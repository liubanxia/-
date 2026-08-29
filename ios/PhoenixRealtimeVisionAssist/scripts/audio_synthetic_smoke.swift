import AudioToolbox
import CoreMedia
import Foundation

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

    var sampleBuffer: CMSampleBuffer?
    let sampleStatus = CMAudioSampleBufferCreateReadyWithPacketDescriptions(
        allocator: kCFAllocatorDefault,
        dataBuffer: blockBuffer,
        formatDescription: formatDescription,
        sampleCount: frames,
        presentationTimeStamp: .zero,
        packetDescriptions: nil,
        sampleBufferOut: &sampleBuffer
    )
    guard sampleStatus == noErr, let sampleBuffer else {
        throw NSError(domain: "AudioSyntheticSmoke", code: Int(sampleStatus), userInfo: [NSLocalizedDescriptionKey: "audio sample buffer failed"])
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
        let analyzer = BroadcastAudioTelemetryAnalyzer()
        analyzer.reset()
        let sampleBuffer = try makeStereoFloat32SampleBuffer()
        analyzer.consume(sampleBuffer)

        guard let snapshot = analyzer.snapshotForTesting() else {
            fatalError("FAIL: CMSampleBuffer was not parsed into telemetry")
        }
        guard snapshot.analysisCount >= 1 else {
            fatalError("FAIL: analysis count did not advance")
        }
        guard snapshot.leftLevel > snapshot.rightLevel else {
            fatalError("FAIL: stereo balance wrong L=\(snapshot.leftLevel) R=\(snapshot.rightLevel)")
        }
        guard snapshot.peakLevel > 0 else {
            fatalError("FAIL: peak remained zero")
        }
        guard Int(snapshot.sampleRate.rounded()) == 48_000 else {
            fatalError("FAIL: sample rate=\(snapshot.sampleRate)")
        }
        guard snapshot.channels == 2 else {
            fatalError("FAIL: channels=\(snapshot.channels)")
        }
        guard snapshot.dominantBand == 2 else {
            fatalError("FAIL: dominant band expected 400 Hz probe (2), got \(snapshot.dominantBand)")
        }

        guard let rawState = analyzer.publishedStateForTesting(), rawState != 0 else {
            fatalError("FAIL: analyzer did not publish Darwin telemetry state")
        }
        let telemetry = decode(rawState)
        guard telemetry.ready, telemetry.active else {
            fatalError("FAIL: packed telemetry not active/ready raw=\(rawState)")
        }
        guard telemetry.count >= 1,
              telemetry.left > telemetry.right,
              telemetry.peak > 0,
              telemetry.band == 2,
              telemetry.sampleRateKHz == 48,
              telemetry.channels == 2 else {
            fatalError("FAIL: packed telemetry mismatch raw=\(rawState) decoded=\(telemetry)")
        }

        print("PASS: synthetic PCM -> audio CMSampleBuffer -> parser -> aggregate snapshot -> Darwin packed telemetry")
        print("count=\(telemetry.count) L=\(telemetry.left) R=\(telemetry.right) peak=\(telemetry.peak) band=\(telemetry.band) rate=\(telemetry.sampleRateKHz)kHz channels=\(telemetry.channels)")
    }
}
