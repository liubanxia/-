import Accelerate
import CoreMedia

final class RealtimeAudioAnalyzer {
    private let lock = NSLock()
    private var smoothedProximity: Double = 0

    var proximity: Double {
        lock.lock()
        defer { lock.unlock() }
        return smoothedProximity
    }

    func consume(sampleBuffer: CMSampleBuffer) {
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
        var rmsValues: [Float] = []
        rmsValues.reserveCapacity(buffers.count)

        for buffer in buffers {
            guard let data = buffer.mData else { continue }
            let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
            guard count > 0 else { continue }
            let samples = data.bindMemory(to: Float.self, capacity: count)
            var rms: Float = 0
            vDSP_rmsqv(samples, 1, &rms, vDSP_Length(count))
            if rms.isFinite { rmsValues.append(rms) }
        }

        guard !rmsValues.isEmpty else { return }
        let rms = rmsValues.reduce(0, +) / Float(rmsValues.count)

        // Relative intensity only. This is intentionally not treated as physical distance.
        let db = 20.0 * log10(max(Double(rms), 0.000_001))
        let normalized = min(max((db + 60.0) / 45.0, 0), 1)

        lock.lock()
        smoothedProximity = smoothedProximity * 0.72 + normalized * 0.28
        lock.unlock()
    }

    func reset() {
        lock.lock()
        smoothedProximity = 0
        lock.unlock()
    }
}
