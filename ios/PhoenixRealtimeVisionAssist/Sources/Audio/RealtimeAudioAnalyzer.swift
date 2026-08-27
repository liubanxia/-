import Accelerate
import CoreMedia
import Foundation

final class RealtimeAudioAnalyzer {
    private let lock = NSLock()
    private var smoothedProximity: Double = 0
    private var lastAnalysisUptime: TimeInterval = 0

    var proximity: Double {
        lock.lock()
        defer { lock.unlock() }
        return smoothedProximity
    }

    func consume(sampleBuffer: CMSampleBuffer) {
        let now = ProcessInfo.processInfo.systemUptime
        let minimumInterval = RuntimeResourcePolicy.audioAnalysisInterval
        guard minimumInterval.isFinite,
              now - lastAnalysisUptime >= minimumInterval else {
            return
        }
        lastAnalysisUptime = now

        guard let format = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee else { return }

        let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
        let isLinearPCM = asbd.mFormatID == kAudioFormatLinearPCM
        guard isLinearPCM, isFloat, asbd.mBitsPerChannel == 32 else { return }

        var retainedBlockBuffer: CMBlockBuffer?
        var bufferList = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(mNumberChannels: 0, mDataByteSize: 0, mData: nil)
        )
        var neededSize = 0

        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &neededSize,
            bufferListOut: &bufferList,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: 0,
            blockBufferOut: &retainedBlockBuffer
        )
        guard status == noErr else { return }

        let buffers = UnsafeMutableAudioBufferListPointer(&bufferList)
        var rmsTotal: Float = 0
        var rmsCount: Int = 0

        for buffer in buffers {
            guard let data = buffer.mData else { continue }
            let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
            guard count > 0 else { continue }
            let samples = data.bindMemory(to: Float.self, capacity: count)
            var rms: Float = 0
            vDSP_rmsqv(samples, 1, &rms, vDSP_Length(count))
            if rms.isFinite {
                rmsTotal += rms
                rmsCount += 1
            }
        }

        guard rmsCount > 0 else { return }
        let rms = rmsTotal / Float(rmsCount)

        // Relative signal intensity only; never interpreted as a physical location.
        let db = 20.0 * log10(max(Double(rms), 0.000_001))
        let normalized = min(max((db + 60.0) / 45.0, 0), 1)

        lock.lock()
        smoothedProximity = smoothedProximity * 0.78 + normalized * 0.22
        lock.unlock()
    }

    func reset() {
        lock.lock()
        smoothedProximity = 0
        lastAnalysisUptime = 0
        lock.unlock()
    }
}
