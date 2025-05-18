using Test 
using SafeTestsets
 
@testset "Utils" verbose=true begin
    @safetestset "linalg" begin include("utils/linalg.jl") end 
end;