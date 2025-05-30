# State Interface

The `AbstractStateEstimate` type defines the interface for representing state 
estimation results in a filtering context. This abstraction enables interoperability across 
filtering algorithms with different internal representations, including parametric and 
nonparametric forms.


```@docs
AbstractStateEstimate
AbstractTimeConstantStateEstimate
StateEstimate
```

## Statistics 

The following functions extract key statistical descriptors from a given state estimate. 
Each function may return exact or empirical quantities, depending on the underlying 
implementation of `AbstractStateEstimate`.

```@docs
estimate(::AbstractStateEstimate)
variance(::AbstractStateEstimate)
covariance(::AbstractStateEstimate)
skewness(::AbstractStateEstimate)
kurtosis(::AbstractStateEstimate)
confidence(::AbstractStateEstimate)
```