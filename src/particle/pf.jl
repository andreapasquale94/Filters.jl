
abstract type AbstractParticleFilter{T} <: AbstractSequentialFilter end

struct ParticleFilter{
    T <: Number,
    S <: AbstractStateEstimate,
    P <: AbstractFilterPrediction,
    R <: AbstractParticleResampling
} <: AbstractParticleFilter{T}
    est::S
    likelihood::P
    resampling::R
end