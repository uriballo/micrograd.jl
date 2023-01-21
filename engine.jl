mutable struct ScalarValue
    data::Float64
    grad::Float64

    backward::Function
    prev::Set{ScalarValue}
    op::String
end

# Base Functions

function Base.:(+)(a::ScalarValue, b)
    c = typeof(b) == typeof(a) ? b : ScalarValue(b, 0.0, nothing, nothing, nothing) 
    
    output = ScalarValue(a.data + c.data, 0.0, nothing, Set([a, c]), "+")

    function backwardf()
        a.grad += output.grad
        c.grad += output.grad
    end

    output.backward = backwardf

    return output
end

function Base.:(-)(a::ScalarValue, b)
    return a + (-b)
end

function Base.:(*)(a::ScalarValue, b)
    c = typeof(b) == typeof(a) ? b : ScalarValue(b, 0.0, nothing, nothing, nothing) 
    
    output = ScalarValue(a.data * c.data, 0.0, nothing, Set([a, c]), "*")

    function backwardf()
        a.grad += c.data * output.grad
        c.grad += a.data * output.grad
    end

    output.backward = backwardf

    return output
end

function Base.:(^)(a::ScalarValue, b::Real)
    output = ScalarValue(a.data ^ b, 0.0, nothing, Set([a]), "^$b")

    function backwardf()
        a.grad += b * a.data ^ (b - 1) * output.grad
    end

    output.backward = backwardf

    return output
end

function Base.:(-)(a::ScalarValue)
    return a * -1
end

function Base.:(/)(a::ScalarValue, b::Real)
    return a * b^(-1)
end

function relu(a::ScalarValue)
    output = a.data < 0 ? 0 : ScalarValue(a.data, 0.0, nothing, Set([a]), "relu")

    function backwardf()
        a.grad += (output.data > 0) * output.grad
    end

end

function backward(a::ScalarValue)
    # Topological order of the children in the graph.
    topo = []

    visited = Set()

    function buildtop(v)
        if v ∉ visited
            push!(visited, v)
            for child in v.prev
                buildtop(child)
            end
            push!(topo, v)
        end
    end
    
    asorted = reverse(buildtop(a))
    
    a.grad = 1
    
    for v in asorted
        backward(v)
    end
end