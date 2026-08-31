import AudioToolbox
import CoreMedia
import Foundation

/// Build 31 stereo spatial cue analyzer.
/// Uses inter-channel delay/coherence plus level bias; it never stores PCM history.
final class BroadcastSpatialAudioAnalyzer {
    private let publisher = SpatialAudioStatePublisher()
    private let lock = NSLock()
    private var lastAnalysisUptime: TimeInterval = 0
    private var priorPeak: Double = 0
    private var active = false

    func reset() {
        lock.lock()
        lastAnalysisUptime = 0
        priorPeak = 0
        active = true
        lock.unlock()
        publisher.clear()
    }

    func finish() {
        lock.lock()
        active = false
        lock.unlock()
        publisher.clear()
    }

    func consume(_ sampleBuffer: CMSampleBuffer) {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        guard now - lastAnalysisUptime >= 0.16 else {
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
              asbd.mChannelsPerFrame >= 2,
              asbd.mSampleRate > 0 else { return }

        let isFloat32 = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
            && asbd.mBitsPerChannel == 32
        let isSigned16 = (asbd.mFormatFlags & kAudioFormatFlagIsSignedInteger) != 0
            && asbd.mBitsPerChannel == 16
        guard isFloat32 || isSigned16 else { return }

        let nonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
        let channels = Int(asbd.mChannelsPerFrame)
        var requiredSize = 0
        var sizingBlock: CMBlockBuffer?
        let sizingStatus = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &requiredSize,
            bufferListOut: nil,
            bufferListSize: 0,
            blockBufferAllocator: kCFAllocatorDefault,
            blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: UInt32(kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment),
            blockBufferOut: &sizingBlock
        )
        guard sizingStatus == noErr || sizingStatus == kCMSampleBufferError_ArrayTooSmall,
              requiredSize >= MemoryLayout<AudioBufferList>.size else { return }

        let rawList = UnsafeMutableRawPointer.allocate(byteCount: requiredSize, alignment: 16)
        rawList.initializeMemory(as: UInt8.self, repeating: 0, count: requiredSize)
        defer { rawList.deallocate() }
        let list = rawList.bindMemory(to: AudioBufferList.self, capacity: 1)

        var confirmedSize = requiredSize
        var retainedBlock: CMBlockBuffer?
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &confirmedSize,
            bufferListOut: list,
            bufferListSize: requiredSize,
            blockBufferAllocator: kCFAllocatorDefault,
            blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: UInt32(kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment),
            blockBufferOut: &retainedBlock
        )
        guard status == noErr else { return }

        let buffers = UnsafeMutableAudioBufferListPointer(list)
        guard !buffers.isEmpty else { return }

        let bytesPerSample = isFloat32 ? MemoryLayout<Float>.size : MemoryLayout<Int16>.size
        let maxFrames = 768
        var left: [Double] = []
        var right: [Double] = []
        left.reserveCapacity(maxFrames)
        right.reserveCapacity(maxFrames)

        if nonInterleaved {
            guard buffers.count >= 2,
                  let leftData = buffers[0].mData,
                  let rightData = buffers[1].mData else { return }
            let frameCount = min(
                maxFrames,
                min(Int(buffers[0].mDataByteSize), Int(buffers[1].mDataByteSize)) / bytesPerSample
            )
            guard frameCount >= 96 else { return }
            for index in 0..<frameCount {
                left.append(readSample(leftData, index: index, isFloat32: isFloat32))
                right.append(readSample(rightData, index: index, isFloat32: isFloat32))
            }
        } else {
            guard let data = buffers[0].mData else { return }
            let frameBytes = bytesPerSample * channels
            let frameCount = min(maxFrames, Int(buffers[0].mDataByteSize) / max(frameBytes, 1))
            guard frameCount >= 96 else { return }
            for frame in 0..<frameCount {
                let base = frame * channels
                left.append(readSample(data, index: base, isFloat32: isFloat32))
                right.append(readSample(data, index: base + 1, isFloat32: isFloat32))
            }
        }

        removeMean(&left)
        removeMean(&right)
        let leftRMS = rms(left)
        let rightRMS = rms(right)
        let peak = max(peakMagnitude(left), peakMagnitude(right))
        let maxLag = min(32, max(6, Int((asbd.mSampleRate * 0.00067).rounded())))
        let correlation = bestNormalizedCorrelation(left: left, right: right, maxLag: maxLag)

        let levelBias = (rightRMS - leftRMS) / max(leftRMS + rightRMS, 0.000_001)
        // Positive lag means the right channel best matches a later sample and therefore the
        // acoustic event reached the left channel first. Convert that to negative lateral.
        let delayBias = -Double(correlation.lag) / Double(max(maxLag, 1))
        let coherence = min(max(correlation.correlation, 0), 1)
        let delayWeight = 0.30 + 0.50 * coherence
        let levelWeight = 1 - delayWeight
        let lateral = min(max(levelBias * levelWeight + delayBias * delayWeight, -1), 1)

        lock.lock()
        let oldPeak = priorPeak
        priorPeak = oldPeak * 0.82 + peak * 0.18
        let isActive = active
        lock.unlock()

        let transient = peak >= 0.028 && peak > max(0.035, oldPeak * 1.45)
        let energy = min(max((max(leftRMS, rightRMS) - 0.003) / 0.12, 0), 1)
        let confidence = min(
            max(coherence * 0.66 + abs(levelBias) * 0.18 + energy * 0.16, 0),
            1
        )

        publisher.publish(
            SharedSpatialAudioEvidence(
                lateral: lateral,
                confidence: confidence,
                coherence: coherence,
                transient: transient,
                active: isActive
            ),
            timestamp: now
        )
    }

    private func bestNormalizedCorrelation(
        left: [Double],
        right: [Double],
        maxLag: Int
    ) -> (lag: Int, correlation: Double) {
        guard left.count == right.count, left.count > maxLag * 2 else { return (0, 0) }
        var bestLag = 0
        var bestValue = -1.0

        for lag in (-maxLag)...maxLag {
            let leftStart = max(0, -lag)
            let rightStart = max(0, lag)
            let count = min(left.count - leftStart, right.count - rightStart)
            guard count >= 64 else { continue }

            var dot = 0.0
            var leftEnergy = 0.0
            var rightEnergy = 0.0
            for offset in 0..<count {
                let l = left[leftStart + offset]
                let r = right[rightStart + offset]
                dot += l * r
                leftEnergy += l * l
                rightEnergy += r * r
            }
            let denominator = sqrt(max(leftEnergy * rightEnergy, 0.000_000_001))
            let value = dot / denominator
            if value > bestValue {
                bestValue = value
                bestLag = lag
            }
        }
        return (bestLag, max(bestValue, 0))
    }

    private func removeMean(_ samples: inout [Double]) {
        guard !samples.isEmpty else { return }
        let mean = samples.reduce(0, +) / Double(samples.count)
        for index in samples.indices { samples[index] -= mean }
    }

    private func rms(_ samples: [Double]) -> Double {
        guard !samples.isEmpty else { return 0 }
        var sum = 0.0
        for value in samples {
            let clipped = min(max(value, -1), 1)
            sum += clipped * clipped
        }
        return sqrt(sum / Double(samples.count))
    }

    private func peakMagnitude(_ samples: [Double]) -> Double {
        samples.reduce(0) { max($0, min(abs($1), 1)) }
    }

    private func readSample(
        _ data: UnsafeMutableRawPointer,
        index: Int,
        isFloat32: Bool
    ) -> Double {
        if isFloat32 {
            let value = data.bindMemory(to: Float.self, capacity: index + 1)[index]
            return value.isFinite ? Double(value) : 0
        }
        return Double(data.bindMemory(to: Int16.self, capacity: index + 1)[index])
            / Double(Int16.max)
    }
}
