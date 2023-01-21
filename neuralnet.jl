include("engine.jl")

mutable struct Neuron
    weights::Array{ScalarValue}
    bias::ScalarValue
    nonlinear::Bool
end

function Neuron(ninputs)
    weights = [ScalarValue(randn(-1,1), 0.0, nothing, nothing, nothing) for _ in 1:ninputs]
    bias = ScalarValue(0, 0.0, nothing, nothing, nothing)
    
    return Neuron(weights, bias, true)
end

function compute(n::Neuron, x)
    act = sum(wᵢ⋅xᵢ for (wᵢ, xᵢ) in zip(n.weights, x)) + n.bias
    return n.nonlinear ? relu(act) : act
end

function params(n::Neuron)
    return n.weights + [n.bias]
end

mutable struct Layer
    neurons::Array{Neuron}
end

function Layer(ninputs, noutputs)
    neurons = [Neuron(ninputs) for _ in 1:noutputs]
    return Layer(neurons)
end

function compute(l::Layer, x)
    return [compute(n, x) for n in l.neurons]
end

function params(l::Layer)
    return [params(n) for n in l.neurons]
end

mutable struct MLP
    layers::Array{Layer}
end

function MLP(ninputs, noutputs)
    # TBD
end
