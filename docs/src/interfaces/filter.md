# Filter Interface

This section defines the abstract types and core methods that form the interface for filters. 
Filters are classified based on their structure (e.g., sequential filters) and provide methods 
for prediction, update, and state estimation. 

```
                      ┌────────────────────┐
                      │   AbstractFilter   │
                      └────────┬───────────┘
                               │
                               ▼
                  ┌─────────────────────────────┐
                  │  AbstractSequentialFilter   │
                  └────────────┬────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   predict!(f; ...)     update!(f, obs; ...)   step!(f, obs; ...)
                               │
                               ▼
                         estimate(f)
                               │
                               ▼
                    ┌───────────────────────┐
                    │ AbstractStateEstimate │
                    └─────────┬─────────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
estimate(state)          skewness(state)        confidence(state)
variance(state)          kurtosis(state)              ...
covariance(state)            ...                      ...
```


## Abstract filters

The most general abstraction for filters.

```@docs
AbstractFilter
init!(::AbstractFilter)
```

## Sequential filters 

Sequential filters process data recursively, performing predict-update cycles as new 
observations become available.


```@docs
AbstractSequentialFilter
predict!(p::AbstractSequentialFilter; kwargs...)
update!(u::AbstractSequentialFilter, obs; kwargs...)
step!(f::AbstractSequentialFilter, obs; kwargs...)
estimate(f::AbstractSequentialFilter)
```

### Prediction and Update Interfaces

These interfaces allow decoupling the filtering logic from the storage or 
representation of the state estimate.

```@docs
AbstractFilterPrediction
predict!(est::AbstractStateEstimate, p::AbstractFilterPrediction; kwargs...)
```

----

```@docs
AbstractFilterUpdate
update!(est::AbstractStateEstimate, u::AbstractFilterUpdate, obs; kwargs...)
```