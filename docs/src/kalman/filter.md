# Generic Kalman Filter interface

```@docs
Filters.BaseKalmanFilter
```

----

```@docs 
init!(::Filters.BaseKalmanFilter)
predict!(::Filters.BaseKalmanFilter{T}; kwargs...) where T
update!(::Filters.BaseKalmanFilter{T}, z::AbstractVector{T}; kwargs...) where T
step!(::Filters.BaseKalmanFilter{T}, z::AbstractVector{T}; kwargs...) where T
estimate(::Filters.BaseKalmanFilter)
```
