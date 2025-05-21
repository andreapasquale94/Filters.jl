using Test 
using SafeTestsets

@testset "Kalman Filters" verbose=true begin 
    @safetestset "LKF" begin include("kalman/kf.jl") end
    @safetestset "LKF" begin include("kalman/ekf.jl") end
end;
 
@testset "Utils" verbose=true begin
    @safetestset "linalg" begin include("utils/linalg.jl") end 
end;