"""
    AbstractKalmanFilter{T}

Abstract type for all Kalman filters implementations.
"""
abstract type AbstractKalmanFilter{T} <: AbstractSequentialFilter{T} end