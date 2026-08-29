import AudioToolbox
import CoreMedia
import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func test_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_get_state")
private func test_notify_get_state(_ token: Int32, _ state: UnsafeMutablePointer<UInt64>) -> UInt32

@_silgen_name("notify_cancel")
private func test_notify_cancel(_ token: Int32) -> UInt32

private func makeStereoFloat32SampleBuffer(
    sampleRate: Double = 48_000,
    frames: Int = 512,
    leftFrequency: Double = 400,
    rightFrequency: Double = 900,
    leftAmplitude: Float = 0.35,
    rightAmplitude: Float = 0.10
) throws -> CMSampleBuffer {
    var asbd = AudioStreamBasicDescription(
        mSampleRate: sampleRate,
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        mBytesPerPacket: UInt32(MemoryLayout<Float>.size * 2),
        mFramesPerPacket: 1,
        mBytesPerFrame: UInt32(MemoryLayout<Float>.size * 2),
        mChannelsPerFrame: 2,
        mBitsPerChannel: 32,
        mReserved: 0
    )

    var formatDescription: CMAudioFormatDescription?
    let formatStatus = CMAudioFormatDescriptionCreate(
        allocator: kCFAllocatorDefault,
        asbd: &asbd,
        layoutSize: 0,
        layout: nil,
        magicCookieSize: 0,
        magicCookie: nil,
        extensions: nil,
        formatDescriptionOut: &formatDescription
    )
    guard formatStatus == noErr, let formatDescription else {
        throw NSError(domain: "AudioSyntheticSmoke", code: Int(formatStatus), userInfo: [NSLocalizedDescriptionKey: "format description failed"])
    }

    var interleaved = [Float](repeating: 0, count: frames * 2)
    for frame in 0..<frames {
        let t = Double(frame) / sampleRate
        interleaved[frame * 2] = leftAmplitude * Float(sin(2 * Double.pi * leftFrequency * t))
        interleaved[frame * 2 + 1] = rightAmplitude * Float(sin(2 * Double.pi * rightFrequency * t))
    }

    let byteCount = interleaved.count * MemoryLayout<Float>.size
    var blockBuffer: CMBlockBuffer?
    let blockStatus = CMBlockBufferCreateWithMemoryBlock(
        allocator: kCFAllocatorDefault,
        memoryBlock: nil,
        blockLength: byteCount,
        blockAllocator: kCFAllocatorDefault,
        customBlockSource: nil,
        offsetToData: 0,
        dataLength: byteCount,
        flags: 0,
        blockBufferOut: &blockBuffer
    )
    guard blockStatus == kCMBlockBufferNoErr, let blockBuffer else {
        throw NSError(domain: "AudioSyntheticSmoke", code: Int(blockStatus), userInfo: [NSLocalizedDescriptionKey: "block buffer failed"])
    }

    let replaceStatus = interleaved.withUnsafeBytes { bytes in
        CMBlockBufferReplaceDataBytes(
            with: bytes.baseAddress!,
            blockBuffer: blockBuffer,
            offsetIntoDestination: 0,
            dataLength: byteCount
        )
    }
    guard replaceStatus == kCMBlockBufferNoErr else {
        throw NSError(domain: "AudioSyntheticSmoke", code: Int(replaceStatus), userInfo: [NSLocalizedDescriptionKey: "copy PCM failed"])
    }

    var timing = CMSampleTimingInfo(
        duration: CMTime(value: 1, timescale: CMTimeScale(sampleRate)),
        presentationTimeStamp: .zero,
        decodeTimeStamp: .invalid
    )
    var sampleBuffer: CMSampleBuffer?
    let sampleStatus = CMSampleBufferCreateReady(
        allocator: kCFAllocatorDefault,
        dataBuffer: blockBuffer,
        formatDescription: formatDescription,
        sampleCount: frames,
        sampleTimingEntryCount: 1,
        sampleTimingArray: &timing,
        sampleSizeEntryCount: 1,
        sampleSizeArray: [MemoryLayout<Float>.size * 2],
        sampleBufferOut: &sampleBuffer
    )
    guard sampleStatus == noErr, let sampleBuffer else {
        throw NSError(domain: "AudioSyntheticSmoke", code: Int(sampleStatus), userInfo: [NSLocalizedDescriptionKey: "sample buffer failed"])
    }
    return sampleBuffer
}

private func decode(_ state: UInt64) -> (count: UInt64, left: UInt64, right: UInt64, peak: UInt64, band: UInt64, sampleRateKHz: UInt64, channels: UInt64, active: Bool, ready: Bool) {
    (
        count: state & 0x0FFF,
        left: (state >> 12) & 0xFF,
        right: (state >> 20) & 0xFF,
        peak: (state >> 28) & 0xFF,
        band: (state >> 36) & 0x07,
        sampleRateKHz: (state >> 40) & 0xFF,
        channels: (state >> 48) & 0xFF,
        active: ((state >> 62) & 1) == 1,
        ready: ((state >> 63) & 1) == 1
    )
}

@main
struct AudioSyntheticSmoke {
    static func main() throws {
        var token: Int32 = -1
        let registerStatus = BroadcastAudioTelemetryAnalyzer.notificationName.withCString {
            test_notify_register_check($0, &token)
        }
        guard registerStatus == 0, token >= 0 else {
            fatalError("FAIL: notify registration status=\(registerStatus)")
        }
        defer { _ = test_notify_cancel(token) }

        let analyzer = BroadcastAudioTelemetryAnalyzer()
        analyzer.reset()
        let sampleBuffer = try makeStereoFloat32SampleBuffer()
        analyzer.consume(sampleBuffer)

        var rawState: UInt64 = 0
        guard test_notify_get_state(token, &rawState) == 0 else {
            fatalError("FAIL: notify state unavailable")
        }

        let telemetry = decode(rawState)
        guard telemetry.ready, telemetry.active else {
            fatalError("FAIL: telemetry not active/ready raw=\(rawState)")
        }
        guard telemetry.count >= 1 else {
            fatalError("FAIL: analysis count did not advance")
        }
        guard telemetry.left > telemetry.right else {
            fatalError("FAIL: stereo balance wrong L=\(telemetry.left) R=\(telemetry.right)")
        }
        guard telemetry.peak > 0 else {
            fatalError("FAIL: peak remained zero")
        }
        guard telemetry.sampleRateKHz == 48 else {
            fatalError("FAIL: sample rate code=\(telemetry.sampleRateKHz)")
        }
        guard telemetry.channels == 2 else {
            fatalError("FAIL: channel code=\(telemetry.channels)")
        }
        guard telemetry.band == 2 else {
            fatalError("FAIL: dominant band expected 400 Hz probe (2), got \(telemetry.band)")
        }

        print("PASS: synthetic ReplayKit-style PCM -> CMSampleBuffer -> analyzer -> Darwin telemetry")
        print("count=\(telemetry.count) L=\(telemetry.left) R=\(telemetry.right) peak=\(telemetry.peak) band=\(telemetry.band) rate=\(telemetry.sampleRateKHz)kHz channels=\(telemetry.channels)")
    }
}
