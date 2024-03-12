//
//  LCMScheduler.swift
//
//
//  Created by Guillermo Cique Fernández on 26/10/23.
//

import CoreML
import Accelerate

/// A scheduler used to compute a de-noised image
///
///  This implementation matches:
///  [Hugging Face Diffusers LCMScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lcm.py)
///
/// This scheduler extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
/// non-Markovian guidance
@available(iOS 16.2, macOS 13.1, *)
public final class LCMScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let finalAlphaCumProd: Float
    public let timeSteps: [Int]

    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []
    public private(set) var randomSource: RandomSource
    
    /// Create a scheduler that uses a pseudo linear multi-step (PLMS)  method
    ///
    /// - Parameters:
    ///   - stepCount: Number of inference steps to schedule
    ///   - originalStepCount: The original number of inference steps
    ///   - trainStepCount: Number of training diffusion steps
    ///   - betaSchedule: Method to schedule betas from betaStart to betaEnd
    ///   - betaStart: The starting value of beta for inference
    ///   - betaEnd: The end value for beta for inference
    /// - Returns: A scheduler ready for its first step
    public init(
        strength: Float? = nil,
        stepCount: Int = 4,
        originalStepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012,
        setAlphaToOne: Bool? = nil,
        randomSource: RandomSource
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = originalStepCount
        
        switch betaSchedule {
        case .linear:
            self.betas = linspace(betaStart, betaEnd, trainStepCount)
        case .scaledLinear:
            self.betas = linspace(pow(betaStart, 0.5), pow(betaEnd, 0.5), trainStepCount).map({ $0 * $0 })
        }

        self.alphas = betas.map({ 1.0 - $0 })
        var alphasCumProd = self.alphas
        for i in 1..<alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i -  1]
        }
        self.alphasCumProd = alphasCumProd
        
        // At every step in ddim, we are looking into the previous alphas_cumprod
        // For the final step, there is no previous alphas_cumprod because we are already at 0
        // `setAlphaToOne` decides whether we set this parameter simply to one or
        // whether we use the final alpha of the "non-previous" one.
        if setAlphaToOne ?? true {
            self.finalAlphaCumProd = 1
        } else {
            self.finalAlphaCumProd = alphasCumProd[0]
        }
        
        let stepRatio = trainStepCount / originalStepCount
        var lcmOriginTimesteps = (1...max(1, Int(Float(originalStepCount) * (strength ?? 1)))).map {
            $0 * stepRatio - 1
        }
        lcmOriginTimesteps.reverse()
        let timestepsIndexes = linspace(0, Float(lcmOriginTimesteps.count), stepCount, endpoint: false)
            .map { Int($0) }
        self.timeSteps = timestepsIndexes
            .map { lcmOriginTimesteps[$0] }

        self.randomSource = randomSource
    }
    
    func getScalingsForBoundaryConditionDiscrete(timeStep t: Int) -> (Double, Double) {
        let sigmaData = 0.5 // Default: 0.5
        let powSigmaData = pow(sigmaData, 2)
        let scaledTimestep = Double(t * 10) // Default timestep_scaling: 10
        let powScaledTimestep = pow(scaledTimestep, 2)
        
        let cSkip = powSigmaData / (pow(scaledTimestep, 2) + powSigmaData)
        let cOut = scaledTimestep / pow((powScaledTimestep + powSigmaData), 0.5)
        return (cSkip, cOut)
    }
    
    public func step(
        output: MLShapedArray<Float32>,
        timeStep t: Int,
        sample: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        
        let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
        
        //  1. get previous step value
        let timeStep = Int(t)
        let prevTimestep = Int(stepIndex < timeSteps.count - 1 ? timeSteps[stepIndex + 1] : t)
        
        // 2. compute alphas, betas
        let alphaProdt = alphasCumProd[timeStep]
        let alphaProdtPrev = prevTimestep >= 0 ? alphasCumProd[prevTimestep] : finalAlphaCumProd
        let betaProdt = 1 - alphaProdt
        let betaProdtPrev = 1 - alphaProdtPrev
        
        let sqrtAlphaProdt = sqrt(alphaProdt)
        let sqrtBetaProdt = sqrt(betaProdt)
        
        // 3. Get scalings for boundary conditions
        let (cSkip, cOut) = getScalingsForBoundaryConditionDiscrete(timeStep: t)
        
        // 4. Compute the predicted original sample x_0 based on the model parameterization
        let predOriginalSample: MLShapedArray<Float32>
        let scalarCount = output.scalarCount
        predOriginalSample = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
            sample.withUnsafeShapedBufferPointer { sample, _, _ in
                output.withUnsafeShapedBufferPointer { output, _, _ in
                    for i in 0..<scalarCount {
                        scalars.initializeElement(at: i, to: (sample[i] - sqrtBetaProdt * output[i]) / sqrtAlphaProdt)
                    }
                }
            }
        }
        
        // 6. Denoise model output using boundary conditions
        let denoised = weightedSum(
            [cOut, cSkip],
            [predOriginalSample, sample]
        )
        
        modelOutputs.removeAll(keepingCapacity: true)
        modelOutputs.append(denoised)
        
        // 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        // Noise is not used on the final timestep of the timestep schedule.
        // This also means that noise is not used for one-step sampling.
        let prevSample: MLShapedArray<Float32>
        if t != timeSteps.last {
            let noise = MLShapedArray<Float32>(converting: randomSource.normalShapedArray(output.shape, mean: 0.0, stdev: 1.0))
            let sqrtAlphaProdtPrev = Double(sqrt(alphaProdtPrev))
            let sqrtBetaProdtPrev = Double(sqrt(betaProdtPrev))
            prevSample = weightedSum(
                [sqrtAlphaProdtPrev, sqrtBetaProdtPrev],
                [denoised, noise]
            )
        } else {
            prevSample = denoised
        }
        
        return prevSample
    }
}
