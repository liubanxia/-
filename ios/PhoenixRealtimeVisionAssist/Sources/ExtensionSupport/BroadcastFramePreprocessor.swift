import Accelerate
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO

struct BroadcastPreprocessedFrame {
    let visionROI: CGRect
    let geometryWidth: Double
    let geometryHeight: Double
    let pixelFormat: OSType
}

enum BroadcastFramePreprocessorError: Error {
    case unsupportedPixelFormat(OSType)
    case invalidROI
    case pixelBufferCreation(OSStatus)
    case missingBaseAddress
    case vImageFailure(vImage_Error)
}

/// Fixed-size, allocation-stable preprocessing for the Broadcast Extension.
///
/// Supported ReplayKit source formats:
/// - BGRA
/// - NV12 full range (420f)
/// - NV12 video range (420v)
///
/// The source ROI is expressed in Vision's oriented, bottom-left coordinate space. The
/// preprocessor maps that ROI back to the raw ReplayKit buffer, scales into one reusable square
/// work buffer with scaleFit letterboxing, applies the EXIF orientation only on that small square,
/// and exposes one reusable BGRA model-input CVPixelBuffer. No full-resolution converted frame is
/// allocated and no per-frame model-input buffer is created.
final class BroadcastFramePreprocessor {
    private struct Point2 {
        let x: Double
        let y: Double
    }

    private struct NormalizedRect {
        let x: Double
        let y: Double
        let width: Double
        let height: Double

        var maxX: Double { x + width }
        var maxY: Double { y + height }
    }

    private struct PixelRect {
        let x: Int
        let y: Int
        let width: Int
        let height: Int
    }

    private final class TempBuffer {
        private var pointer: UnsafeMutableRawPointer?
        private var capacity = 0

        deinit {
            pointer?.deallocate()
        }

        func ensure(byteCount: Int) -> UnsafeMutableRawPointer? {
            guard byteCount > 0 else { return nil }
            if byteCount > capacity {
                pointer?.deallocate()
                pointer = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 64)
                capacity = byteCount
            }
            return pointer
        }
    }

    let modelInput: CVPixelBuffer

    private let side: Int
    private let work: CVPixelBuffer
    private let scratch: CVPixelBuffer
    private let yScaled: UnsafeMutableRawPointer
    private let cbcrScaled: UnsafeMutableRawPointer
    private let bgraScaleTemp = TempBuffer()
    private let yScaleTemp = TempBuffer()
    private let cbcrScaleTemp = TempBuffer()
    private var fullConversion = vImage_YpCbCrToARGB()
    private var videoConversion = vImage_YpCbCrToARGB()

    init(side: Int) throws {
        guard side >= 64, side <= 512, side % 2 == 0 else {
            throw BroadcastFramePreprocessorError.invalidROI
        }
        self.side = side
        work = try Self.makeBGRA(width: side, height: side)
        modelInput = try Self.makeBGRA(width: side, height: side)
        scratch = try Self.makeBGRA(width: side, height: side)
        yScaled = UnsafeMutableRawPointer.allocate(byteCount: side * side, alignment: 64)
        cbcrScaled = UnsafeMutableRawPointer.allocate(byteCount: (side / 2) * (side / 2) * 2, alignment: 64)

        var fullRange = vImage_YpCbCrPixelRange(
            Yp_bias: 0,
            CbCr_bias: 128,
            YpRangeMax: 255,
            CbCrRangeMax: 255,
            YpMax: 255,
            YpMin: 0,
            CbCrMax: 255,
            CbCrMin: 0
        )
        var videoRange = vImage_YpCbCrPixelRange(
            Yp_bias: 16,
            CbCr_bias: 128,
            YpRangeMax: 235,
            CbCrRangeMax: 240,
            YpMax: 235,
            YpMin: 16,
            CbCrMax: 240,
            CbCrMin: 16
        )
        let fullError = vImageConvert_YpCbCrToARGB_GenerateConversion(
            kvImage_YpCbCrToARGBMatrix_ITU_R_709_2,
            &fullRange,
            &fullConversion,
            kvImage420Yp8_CbCr8,
            kvImageARGB8888,
            vImage_Flags(kvImageNoFlags)
        )
        let videoError = vImageConvert_YpCbCrToARGB_GenerateConversion(
            kvImage_YpCbCrToARGBMatrix_ITU_R_709_2,
            &videoRange,
            &videoConversion,
            kvImage420Yp8_CbCr8,
            kvImageARGB8888,
            vImage_Flags(kvImageNoFlags)
        )
        guard fullError == kvImageNoError else {
            throw BroadcastFramePreprocessorError.vImageFailure(fullError)
        }
        guard videoError == kvImageNoError else {
            throw BroadcastFramePreprocessorError.vImageFailure(videoError)
        }
    }

    deinit {
        yScaled.deallocate()
        cbcrScaled.deallocate()
    }

    func preprocess(
        source: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        visionROI requestedROI: CGRect
    ) throws -> BroadcastPreprocessedFrame {
        guard let roi = Self.validatedROI(requestedROI) else {
            throw BroadcastFramePreprocessorError.invalidROI
        }

        let pixelFormat = CVPixelBufferGetPixelFormatType(source)
        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        guard sourceWidth > 1, sourceHeight > 1 else {
            throw BroadcastFramePreprocessorError.invalidROI
        }

        let rawNormalized = Self.rawNormalizedRect(forVisionROI: roi, orientation: orientation)
        let rawPixelROI: PixelRect

        switch pixelFormat {
        case kCVPixelFormatType_32BGRA:
            rawPixelROI = Self.pixelRect(
                rawNormalized,
                width: sourceWidth,
                height: sourceHeight,
                requireEvenChromaAlignment: false
            )
            try scaleBGRAIntoWork(source: source, rawROI: rawPixelROI)
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
             kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            rawPixelROI = Self.pixelRect(
                rawNormalized,
                width: sourceWidth,
                height: sourceHeight,
                requireEvenChromaAlignment: true
            )
            try scaleNV12IntoWork(
                source: source,
                rawROI: rawPixelROI,
                fullRange: pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            )
        default:
            throw BroadcastFramePreprocessorError.unsupportedPixelFormat(pixelFormat)
        }

        try orientWorkToModelInput(orientation)

        let effectiveROI = Self.visionROI(
            forRawPixelRect: rawPixelROI,
            rawWidth: sourceWidth,
            rawHeight: sourceHeight,
            orientation: orientation
        )
        let rotatesDimensions = Self.rotatesDimensions(orientation)
        let geometryWidth = Double(rotatesDimensions ? rawPixelROI.height : rawPixelROI.width)
        let geometryHeight = Double(rotatesDimensions ? rawPixelROI.width : rawPixelROI.height)

        return .init(
            visionROI: effectiveROI,
            geometryWidth: geometryWidth,
            geometryHeight: geometryHeight,
            pixelFormat: pixelFormat
        )
    }

    private func scaleBGRAIntoWork(source: CVPixelBuffer, rawROI: PixelRect) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(work, [])
        defer {
            CVPixelBufferUnlockBaseAddress(work, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }

        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let workBase = CVPixelBufferGetBaseAddress(work) else {
            throw BroadcastFramePreprocessorError.missingBaseAddress
        }

        fillLetterbox(workBase)
        let sourceRowBytes = CVPixelBufferGetBytesPerRow(source)
        let workRowBytes = CVPixelBufferGetBytesPerRow(work)
        let fitted = fittedSize(width: rawROI.width, height: rawROI.height)
        let destinationX = (side - fitted.width) / 2
        let destinationY = (side - fitted.height) / 2

        var src = vImage_Buffer(
            data: sourceBase.advanced(by: rawROI.y * sourceRowBytes + rawROI.x * 4),
            height: vImagePixelCount(rawROI.height),
            width: vImagePixelCount(rawROI.width),
            rowBytes: sourceRowBytes
        )
        var dst = vImage_Buffer(
            data: workBase.advanced(by: destinationY * workRowBytes + destinationX * 4),
            height: vImagePixelCount(fitted.height),
            width: vImagePixelCount(fitted.width),
            rowBytes: workRowBytes
        )

        let query = vImageScale_ARGB8888(
            &src,
            &dst,
            nil,
            vImage_Flags(kvImageHighQualityResampling | kvImageGetTempBufferSize)
        )
        let temp = bgraScaleTemp.ensure(byteCount: max(1, Int(query)))
        let error = vImageScale_ARGB8888(
            &src,
            &dst,
            temp,
            vImage_Flags(kvImageHighQualityResampling)
        )
        guard error == kvImageNoError else {
            throw BroadcastFramePreprocessorError.vImageFailure(error)
        }
    }

    private func scaleNV12IntoWork(
        source: CVPixelBuffer,
        rawROI: PixelRect,
        fullRange: Bool
    ) throws {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(work, [])
        defer {
            CVPixelBufferUnlockBaseAddress(work, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }

        guard CVPixelBufferIsPlanar(source), CVPixelBufferGetPlaneCount(source) >= 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(source, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(source, 1),
              let workBase = CVPixelBufferGetBaseAddress(work) else {
            throw BroadcastFramePreprocessorError.missingBaseAddress
        }

        fillLetterbox(workBase)
        let yRowBytes = CVPixelBufferGetBytesPerRowOfPlane(source, 0)
        let uvRowBytes = CVPixelBufferGetBytesPerRowOfPlane(source, 1)
        let workRowBytes = CVPixelBufferGetBytesPerRow(work)
        var fitted = fittedSize(width: rawROI.width, height: rawROI.height)
        fitted.width = max(2, fitted.width & ~1)
        fitted.height = max(2, fitted.height & ~1)
        let destinationX = (side - fitted.width) / 2
        let destinationY = (side - fitted.height) / 2

        var ySource = vImage_Buffer(
            data: yBase.advanced(by: rawROI.y * yRowBytes + rawROI.x),
            height: vImagePixelCount(rawROI.height),
            width: vImagePixelCount(rawROI.width),
            rowBytes: yRowBytes
        )
        var yDestination = vImage_Buffer(
            data: yScaled,
            height: vImagePixelCount(fitted.height),
            width: vImagePixelCount(fitted.width),
            rowBytes: side
        )
        let yQuery = vImageScale_Planar8(
            &ySource,
            &yDestination,
            nil,
            vImage_Flags(kvImageHighQualityResampling | kvImageGetTempBufferSize)
        )
        let yTemp = yScaleTemp.ensure(byteCount: max(1, Int(yQuery)))
        let yError = vImageScale_Planar8(
            &ySource,
            &yDestination,
            yTemp,
            vImage_Flags(kvImageHighQualityResampling)
        )
        guard yError == kvImageNoError else {
            throw BroadcastFramePreprocessorError.vImageFailure(yError)
        }

        var uvSource = vImage_Buffer(
            data: uvBase.advanced(by: (rawROI.y / 2) * uvRowBytes + rawROI.x),
            height: vImagePixelCount(rawROI.height / 2),
            width: vImagePixelCount(rawROI.width / 2),
            rowBytes: uvRowBytes
        )
        var uvDestination = vImage_Buffer(
            data: cbcrScaled,
            height: vImagePixelCount(fitted.height / 2),
            width: vImagePixelCount(fitted.width / 2),
            rowBytes: side
        )
        let uvQuery = vImageScale_CbCr8(
            &uvSource,
            &uvDestination,
            nil,
            vImage_Flags(kvImageHighQualityResampling | kvImageGetTempBufferSize)
        )
        let uvTemp = cbcrScaleTemp.ensure(byteCount: max(1, Int(uvQuery)))
        let uvError = vImageScale_CbCr8(
            &uvSource,
            &uvDestination,
            uvTemp,
            vImage_Flags(kvImageHighQualityResampling)
        )
        guard uvError == kvImageNoError else {
            throw BroadcastFramePreprocessorError.vImageFailure(uvError)
        }

        var destination = vImage_Buffer(
            data: workBase.advanced(by: destinationY * workRowBytes + destinationX * 4),
            height: vImagePixelCount(fitted.height),
            width: vImagePixelCount(fitted.width),
            rowBytes: workRowBytes
        )
        let conversionError: vImage_Error
        if fullRange {
            conversionError = vImageConvert_420Yp8_CbCr8ToARGB8888(
                &yDestination,
                &uvDestination,
                &destination,
                &fullConversion,
                [3, 2, 1, 0],
                255,
                vImage_Flags(kvImageNoFlags)
            )
        } else {
            conversionError = vImageConvert_420Yp8_CbCr8ToARGB8888(
                &yDestination,
                &uvDestination,
                &destination,
                &videoConversion,
                [3, 2, 1, 0],
                255,
                vImage_Flags(kvImageNoFlags)
            )
        }
        guard conversionError == kvImageNoError else {
            throw BroadcastFramePreprocessorError.vImageFailure(conversionError)
        }
    }

    private func orientWorkToModelInput(_ orientation: CGImagePropertyOrientation) throws {
        CVPixelBufferLockBaseAddress(work, .readOnly)
        CVPixelBufferLockBaseAddress(modelInput, [])
        CVPixelBufferLockBaseAddress(scratch, [])
        defer {
            CVPixelBufferUnlockBaseAddress(scratch, [])
            CVPixelBufferUnlockBaseAddress(modelInput, [])
            CVPixelBufferUnlockBaseAddress(work, .readOnly)
        }

        guard let workBase = CVPixelBufferGetBaseAddress(work),
              let modelBase = CVPixelBufferGetBaseAddress(modelInput),
              let scratchBase = CVPixelBufferGetBaseAddress(scratch) else {
            throw BroadcastFramePreprocessorError.missingBaseAddress
        }

        var src = vImage_Buffer(
            data: workBase,
            height: vImagePixelCount(side),
            width: vImagePixelCount(side),
            rowBytes: CVPixelBufferGetBytesPerRow(work)
        )
        var dst = vImage_Buffer(
            data: modelBase,
            height: vImagePixelCount(side),
            width: vImagePixelCount(side),
            rowBytes: CVPixelBufferGetBytesPerRow(modelInput)
        )
        var tmp = vImage_Buffer(
            data: scratchBase,
            height: vImagePixelCount(side),
            width: vImagePixelCount(side),
            rowBytes: CVPixelBufferGetBytesPerRow(scratch)
        )
        let flags = vImage_Flags(kvImageNoFlags)
        var error: vImage_Error

        switch orientation {
        case .up:
            error = vImageRotate90_ARGB8888(
                &src, &dst, UInt8(kRotate0DegreesClockwise), [0, 0, 0, 0], flags
            )
        case .upMirrored:
            error = vImageHorizontalReflect_ARGB8888(&src, &dst, flags)
        case .down:
            error = vImageRotate90_ARGB8888(
                &src, &dst, UInt8(kRotate180DegreesClockwise), [0, 0, 0, 0], flags
            )
        case .downMirrored:
            error = vImageVerticalReflect_ARGB8888(&src, &dst, flags)
        case .right:
            error = vImageRotate90_ARGB8888(
                &src, &dst, UInt8(kRotate90DegreesClockwise), [0, 0, 0, 0], flags
            )
        case .left:
            error = vImageRotate90_ARGB8888(
                &src, &dst, UInt8(kRotate270DegreesClockwise), [0, 0, 0, 0], flags
            )
        case .leftMirrored:
            error = vImageRotate90_ARGB8888(
                &src, &tmp, UInt8(kRotate90DegreesClockwise), [0, 0, 0, 0], flags
            )
            if error == kvImageNoError {
                error = vImageHorizontalReflect_ARGB8888(&tmp, &dst, flags)
            }
        case .rightMirrored:
            error = vImageRotate90_ARGB8888(
                &src, &tmp, UInt8(kRotate90DegreesClockwise), [0, 0, 0, 0], flags
            )
            if error == kvImageNoError {
                error = vImageVerticalReflect_ARGB8888(&tmp, &dst, flags)
            }
        @unknown default:
            error = vImageRotate90_ARGB8888(
                &src, &dst, UInt8(kRotate0DegreesClockwise), [0, 0, 0, 0], flags
            )
        }

        guard error == kvImageNoError else {
            throw BroadcastFramePreprocessorError.vImageFailure(error)
        }
    }

    private func fittedSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let scale = min(Double(side) / Double(width), Double(side) / Double(height))
        let width = max(1, min(side, Int((Double(width) * scale).rounded(.toNearestOrAwayFromZero))))
        let height = max(1, min(side, Int((Double(height) * scale).rounded(.toNearestOrAwayFromZero))))
        return (width, height)
    }

    private func fillLetterbox(_ base: UnsafeMutableRawPointer) {
        let rowBytes = CVPixelBufferGetBytesPerRow(work)
        for y in 0..<side {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
            for x in 0..<side {
                let offset = x * 4
                row[offset] = 114
                row[offset + 1] = 114
                row[offset + 2] = 114
                row[offset + 3] = 255
            }
        }
    }

    private static func makeBGRA(width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &buffer
        )
        guard status == kCVReturnSuccess, let buffer else {
            throw BroadcastFramePreprocessorError.pixelBufferCreation(status)
        }
        return buffer
    }

    private static func validatedROI(_ requested: CGRect) -> CGRect? {
        guard requested.origin.x.isFinite,
              requested.origin.y.isFinite,
              requested.size.width.isFinite,
              requested.size.height.isFinite else { return nil }
        let unit = CGRect(x: 0, y: 0, width: 1, height: 1)
        let roi = requested.standardized.intersection(unit)
        guard !roi.isNull, roi.width > 0.01, roi.height > 0.01 else { return nil }
        return roi
    }

    private static func rawToOriented(
        _ point: Point2,
        orientation: CGImagePropertyOrientation
    ) -> Point2 {
        switch orientation {
        case .up: return point
        case .upMirrored: return .init(x: 1 - point.x, y: point.y)
        case .down: return .init(x: 1 - point.x, y: 1 - point.y)
        case .downMirrored: return .init(x: point.x, y: 1 - point.y)
        case .leftMirrored: return .init(x: point.y, y: point.x)
        case .right: return .init(x: 1 - point.y, y: point.x)
        case .rightMirrored: return .init(x: 1 - point.y, y: 1 - point.x)
        case .left: return .init(x: point.y, y: 1 - point.x)
        @unknown default: return point
        }
    }

    private static func orientedToRaw(
        _ point: Point2,
        orientation: CGImagePropertyOrientation
    ) -> Point2 {
        switch orientation {
        case .up: return point
        case .upMirrored: return .init(x: 1 - point.x, y: point.y)
        case .down: return .init(x: 1 - point.x, y: 1 - point.y)
        case .downMirrored: return .init(x: point.x, y: 1 - point.y)
        case .leftMirrored: return .init(x: point.y, y: point.x)
        case .right: return .init(x: point.y, y: 1 - point.x)
        case .rightMirrored: return .init(x: 1 - point.y, y: 1 - point.x)
        case .left: return .init(x: 1 - point.y, y: point.x)
        @unknown default: return point
        }
    }

    private static func rawNormalizedRect(
        forVisionROI roi: CGRect,
        orientation: CGImagePropertyOrientation
    ) -> NormalizedRect {
        let oriented = NormalizedRect(
            x: Double(roi.minX),
            y: Double(1 - roi.maxY),
            width: Double(roi.width),
            height: Double(roi.height)
        )
        let corners = [
            Point2(x: oriented.x, y: oriented.y),
            Point2(x: oriented.maxX, y: oriented.y),
            Point2(x: oriented.x, y: oriented.maxY),
            Point2(x: oriented.maxX, y: oriented.maxY),
        ].map { orientedToRaw($0, orientation: orientation) }

        let minX = corners.map(\.x).min() ?? 0
        let maxX = corners.map(\.x).max() ?? 1
        let minY = corners.map(\.y).min() ?? 0
        let maxY = corners.map(\.y).max() ?? 1
        return .init(
            x: min(max(minX, 0), 1),
            y: min(max(minY, 0), 1),
            width: min(max(maxX - minX, 0), 1),
            height: min(max(maxY - minY, 0), 1)
        )
    }

    private static func pixelRect(
        _ normalized: NormalizedRect,
        width: Int,
        height: Int,
        requireEvenChromaAlignment: Bool
    ) -> PixelRect {
        var x0 = Int(floor(normalized.x * Double(width)))
        var y0 = Int(floor(normalized.y * Double(height)))
        var x1 = Int(ceil(normalized.maxX * Double(width)))
        var y1 = Int(ceil(normalized.maxY * Double(height)))

        if requireEvenChromaAlignment {
            x0 &= ~1
            y0 &= ~1
            x1 = (x1 + 1) & ~1
            y1 = (y1 + 1) & ~1
        }

        x0 = max(0, min(width - (requireEvenChromaAlignment ? 2 : 1), x0))
        y0 = max(0, min(height - (requireEvenChromaAlignment ? 2 : 1), y0))
        x1 = max(x0 + (requireEvenChromaAlignment ? 2 : 1), min(width, x1))
        y1 = max(y0 + (requireEvenChromaAlignment ? 2 : 1), min(height, y1))

        if requireEvenChromaAlignment {
            if x1 % 2 != 0 { x1 -= 1 }
            if y1 % 2 != 0 { y1 -= 1 }
            if x1 <= x0 { x1 = min(width, x0 + 2) }
            if y1 <= y0 { y1 = min(height, y0 + 2) }
        }

        return .init(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }

    private static func visionROI(
        forRawPixelRect raw: PixelRect,
        rawWidth: Int,
        rawHeight: Int,
        orientation: CGImagePropertyOrientation
    ) -> CGRect {
        let rawRect = NormalizedRect(
            x: Double(raw.x) / Double(rawWidth),
            y: Double(raw.y) / Double(rawHeight),
            width: Double(raw.width) / Double(rawWidth),
            height: Double(raw.height) / Double(rawHeight)
        )
        let corners = [
            Point2(x: rawRect.x, y: rawRect.y),
            Point2(x: rawRect.maxX, y: rawRect.y),
            Point2(x: rawRect.x, y: rawRect.maxY),
            Point2(x: rawRect.maxX, y: rawRect.maxY),
        ].map { rawToOriented($0, orientation: orientation) }

        let minX = min(max(corners.map(\.x).min() ?? 0, 0), 1)
        let maxX = min(max(corners.map(\.x).max() ?? 1, 0), 1)
        let minYTop = min(max(corners.map(\.y).min() ?? 0, 0), 1)
        let maxYTop = min(max(corners.map(\.y).max() ?? 1, 0), 1)
        return CGRect(
            x: minX,
            y: 1 - maxYTop,
            width: max(0.001, maxX - minX),
            height: max(0.001, maxYTop - minYTop)
        ).intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    private static func rotatesDimensions(_ orientation: CGImagePropertyOrientation) -> Bool {
        switch orientation {
        case .left, .leftMirrored, .right, .rightMirrored:
            return true
        default:
            return false
        }
    }
}
