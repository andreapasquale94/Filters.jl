# ——— State ————————————————————————————————————————————————————————————————————————————————

"""
    LTIStateModel{T}

Linear time-invariant state model storing the system matrix `F` and the input matrix `B`.
"""
struct LTIStateModel{T <: Number} <: AbstractTimeConstantStateModel
    F::Matrix{T}  # F * x 
    B::Matrix{T}  # B * u 
end

function propagate!(m::LTIStateModel, xn, x; u = missing, kwargs...)
    @inbounds begin
        xn .= m.F * x
        if !ismissing(u)
            xn .+= m.B * u
        end
    end
    nothing
end

@inline stm(m::LTIStateModel) = m.F

# ——— Observation ——————————————————————————————————————————————————————————————————————————

"""
    LTIObservationModel{T}

Linear time-invariant observation model storing the output matrix `H` and the feed-forward
matrix `D`.
"""
struct LTIObservationModel{T <: Number} <: AbstractTimeConstantObservationModel
    H::Matrix{T}  # H * x 
    D::Matrix{T}  # D * u 
end

function observe!(m::LTIObservationModel, z, x; u = missing, kwargs...)
    @inbounds begin
        z .= m.H * x
        if !ismissing(u)
            z .+= m.D * u
        end
    end
    nothing
end

@inline ojac(m::LTIObservationModel) = m.H