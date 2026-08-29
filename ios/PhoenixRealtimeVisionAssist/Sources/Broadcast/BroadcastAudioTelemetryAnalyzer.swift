import AudioToolbox
import CoreMedia
import Darwin
import Foundation

@_silgen_name("notify_register_check")
private func liteview_audio_notify_register_check(
    _ name: UnsafePointer<CChar>,
    _ token: UnsafeMutablePointer<Int32>
) -> UInt32

@_silgen_name("notify_set_state")
private func liteview_audio_notify_set_state(_ token: Int32, _ state: UInt64) -> UInt32

@_silgen_name("notify_get_state")
private func liteview_audio_notify_get_state(_ token: Int32, _ state: UnsafeMutablePointer<UInt64>) -> UInt32

@_silgen_name("notify_post")
private func liteview_audio_notify_post(_ name: UnsafePointer<CChar>) -> UInt32

@_silgen_name("notify_cancel")
private func liteview_audio_notify_cancel(_ token: Int32) -> UInt32

/// Privacy-preserving aggregate diagnostics for ReplayKit `.audioApp` PCM.
/// No PCM history or audio files are retained.
final class BroadcastAudioTelemetryAnalyzer {
    struct Snapshot: Equatable {
        let analysisCount: UInt64
        let leftLevel: Double
        let rightLevel: Double
        let peakLevel: Double
        let dominantBand: Int
        let transient: Bool
        let sampleRate: Double
        let channels: Int
    }

    static let notificationName = "com.phoenix.realtimevisionassist.broadcast.audio-diagnostics.v1"

    private let lock = NSLock()
    private var token: Int32 = -1
    private var lastAnalysisUptime: TimeInterval = 0
    private var analysisCount: UInt64 = 0
    private var smoothedPeak: Double = 0
    private var lastSnapshotStorage: Snapshot?

    init() {
        var newToken: Int32 = -1
        let status = Self.notificationName.withCString {
            liteview_audio_notify_register_check($0, &newToken)
        }
        if status == 0 { token = newToken }
    }

    deinit {
        if token >= 0 { _ = liteview_audio_notify_cancel(token) }
    }

    func reset() {
        lock.lock()
        lastAnalysisUptime = 0
        analysisCount = 0
        smoothedPeak = 0
        lastSnapshotStorage = nil
        lock.unlock()
        clearPublishedState()
    }

    func finish() {
        clearPublishedState()
    }

    func consume(_ sampleBuffer: CMSampleBuffer) {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        guard now - lastAnalysisUptime >= 0.18 else {
            lock.unlock()
            return
        }
        lastAnalysisUptime = now
        lock.unlock()

        guard let format = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPointer = CMAudioFormatDescriptionGetStreamBasicDescription(format) else {
            return
        }

        let asbd = asbdPointer.pointee
        guard asbd.mFormatID == kAudioFormatLinearPCM,
              asbd.mChannelsPerFrame > 0,
              asbd.mSampleRate > 0 else {
            return
        }

        let channels = max(1, Int(asbd.mChannelsPerFrame))
        let nonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
        let isFloat32 = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0 && asbd.mBitsPerChannel == 32
        let isSigned16 = (asbd.mFormatFlags & kAudioFormatFlagIsSignedInteger) != 0 && asbd.mBitsPerChannel == 16
        guard isFloat32 || isSigned16 else { return }

        // CoreMedia decides the required AudioBufferList size. Do not infer it from Swift
        // MemoryLayout; stereo/non-interleaved layouts can otherwise return -12737
        // (kCMSampleBufferError_ArrayTooSmall).
        var requiredSize = 0
        var sizingBlockBuffer: CMBlockBuffer?
        let sizingStatus = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &requiredSize,
            bufferListOut: nil,
            bufferListSize: 0,
            blockBufferAllocator: kCFAllocatorDefault,
            blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: UInt32(kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment),
            blockBufferOut: &sizingBlockBuffer
        )
        guard sizingStatus == noErr || sizingStatus == kCMSampleBufferError_ArrayTooSmall,
              requiredSize >= MemoryLayout<AudioBufferList>.size else {
            return
        }

        let rawList = UnsafeMutableRawPointer.allocate(byteCount: requiredSize, alignment: 16)
        rawList.initializeMemory(as: UInt8.self, repeating: 0, count: requiredSize)
        defer { rawList.deallocate() }
        let audioBufferList = rawList.bindMemory(to: AudioBufferList.self, capacity: 1)

        var confirmedSize = requiredSize
        var retainedBlockBuffer: CMBlockBuffer?
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &confirmedSize,
            bufferListOut: audioBufferList,
            bufferListSize: requiredSize,
            blockBufferAllocator: kCFAllocatorDefault,
            blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: UInt32(kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment),
            blockBufferOut: &retainedBlockBuffer
        )
        guard status == noErr else { return }

        let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
        guard !buffers.isEmpty else { return }

        let maxFrames = 512
        var leftSamples: [Double] = []
        var rightSamples: [Double] = []
        leftSamples.reserveCapacity(maxFrames)
        rightSamples.reserveCapacity(maxFrames)

        if nonInterleaved {
            guard let first = buffers.first, let firstData = first.mData else { return }
            let bytesPerSample = isFloat32 ? MemoryLayout<Float>.size : MemoryLayout<Int16>.size
            let frameCount = min(maxFrames, Int(first.mDataByteSize) / bytesPerSample)
            guard frameCount > 0 else { return }

            let rightBuffer = buffers.count > 1 ? buffers[1] : first
            guard let rightData = rightBuffer.mData else { return }
            for frame in 0..<frameCount {
                leftSamples.append(readSample(firstData, index: frame, isFloat32: isFloat32))
                rightSamples.append(readSample(rightData, index: frame, isFloat32: isFloat32))
            }
        } else {
            guard let first = buffers.first, let data = first.mData else { return }
            let bytesPerSample = isFloat32 ? MemoryLayout<Float>.size : MemoryLayout<Int16>.size
            let frameBytes = bytesPerSample * channels
            let frameCount = min(maxFrames, Int(first.mDataByteSize) / max(frameBytes, bytesPerSample))
            guard frameCount > 0 else { return }

            for frame in 0..<frameCount {
                let baseIndex = frame * channels
                let left = readSample(data, index: baseIndex, isFloat32: isFloat32)
                let right = channels > 1
                    ? readSample(data, index: baseIndex + 1, isFloat32: isFloat32)
                    : left
                leftSamples.append(left)
                rightSamples.append(right)
            }
        }

        guard !leftSamples.isEmpty else { return }
        let leftRMS = rms(leftSamples)
        let rightRMS = rms(rightSamples)
        let peak = max(peakMagnitude(leftSamples), peakMagnitude(rightSamples))
        let mono = zip(leftSamples, rightSamples).map { ($0 + $1) * 0.5 }
        let dominantBand = coarseDominantBand(samples: mono, sampleRate: asbd.mSampleRate)

        lock.lock()
        analysisCount &+= 1
        let priorPeak = smoothedPeak
        smoothedPeak = priorPeak * 0.84 + peak * 0.16
        let snapshot = Snapshot(
            analysisCount: analysisCount,
            leftLevel: normalizedLevel(leftRMS),
            rightLevel: normalizedLevel(rightRMS),
            peakLevel: normalizedLevel(peak),
            dominantBand: dominantBand,
            transient: peak >= 0.035 && peak > max(0.045, priorPeak * 1.75),
            sampleRate: asbd.mSampleRate,
            channels: channels
        )
        lastSnapshotStorage = snapshot
        lock.unlock()

        publish(snapshot)
    }

    func snapshotForTesting() -> Snapshot? {
        lock.lock()
        defer { lock.unlock() }
        return lastSnapshotStorage
    }

    func publishedStateForTesting() -> UInt64? {
        guard token >= 0 else { return nil }
        var state: UInt64 = 0
        guard liteview_audio_notify_get_state(token, &state) == 0 else { return nil }
        return state
    }

    private func readSample(_ data: UnsafeMutableRawPointer, index: Int, isFloat32: Bool) -> Double {
        if isFloat32 {
            let value = data.bindMemory(to: Float.self, capacity: index + 1)[index]
            return value.isFinite ? Double(value) : 0
        }
        return Double(data.bindMemory(to: Int16.self, capacity: index + 1)[index]) / Double(Int16.max)
    }

    private func rms(_ samples: [Double]) -> Double {
        guard !samples.isEmpty else { return 0 }
        var sum = 0.0
        for sample in samples {
            let clipped = min(max(sample, -1), 1)
            sum += clipped * clipped
        }
        return sqrt(sum / Double(samples.count))
    }

    private func peakMagnitude(_ samples: [Double]) -> Double {
        samples.reduce(0) { max($0, min(abs($1), 1)) }
    }

    private func normalizedLevel(_ amplitude: Double) -> Double {
        let db = 20 * log10(max(amplitude, 0.000_001))
        return min(max((db + 60) / 60, 0), 1)
    }

    private func coarseDominantBand(samples: [Double], sampleRate: Double) -> Int {
        guard samples.count >= 32 else { return 0 }
        let frequencies: [Double] = [160, 400, 900, 1_800, 3_500, 7_000]
        var bestBand = 0
        var bestEnergy = -Double.infinity
        for (index, frequency) in frequencies.enumerated() where frequency < sampleRate * 0.46 {
            let omega = 2 * Double.pi * frequency / sampleRate
            var real = 0.0
            var imaginary = 0.0
            for sampleIndex in samples.indices {
                let angle = omega * Double(sampleIndex)
                real += samples[sampleIndex] * cos(angle)
                imaginary -= samples[sampleIndex] * sin(angle)
            }
            let energy = real * real + imaginary * imaginary
            if energy > bestEnergy {
                bestEnergy = energy
                bestBand = index + 1
            }
        }
        return bestBand
    }

    private func publish(_ snapshot: Snapshot) {
        guard token >= 0 else { return }
        let state = Self.pack(snapshot)
        guard liteview_audio_notify_set_state(token, state) == 0 else { return }
        _ = Self.notificationName.withCString { liteview_audio_notify_post($0) }
    }

    static func pack(_ snapshot: Snapshot) -> UInt64 {
        let countCode = min(snapshot.analysisCount, 0x0FFF)
        let leftCode = UInt64((min(max(snapshot.leftLevel, 0), 1) * 255).rounded())
        let rightCode = UInt64((min(max(snapshot.rightLevel, 0), 1) * 255).rounded())
        let peakCode = UInt64((min(max(snapshot.peakLevel, 0), 1) * 255).rounded())
        let bandCode = UInt64(min(max(snapshot.dominantBand, 0), 7))
        let sampleRateCode = UInt64(min(max(Int((snapshot.sampleRate / 1_000).rounded()), 0), 255))
        let channelCode = UInt64(min(max(snapshot.channels, 0), 255))

        var state = countCode
        state |= leftCode << 12
        state |= rightCode << 20
        state |= peakCode << 28
        state |= bandCode << 36
        if snapshot.transient { state |= UInt64(1) << 39 }
        state |= sampleRateCode << 40
        state |= channelCode << 48
        state |= UInt64(1) << 62
        state |= UInt64(1) << 63
        return state
    }

    private func clearPublishedState() {
        guard token >= 0 else { return }
        _ = liteview_audio_notify_set_state(token, 0)
        _ = Self.notificationName.withCString { liteview_audio_notify_post($0) }
    }
}
