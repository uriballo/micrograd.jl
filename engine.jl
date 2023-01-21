mutable struct ScalarValue
    data::Float64 # Scalar value
    grad::Float64 # Gradient value
    backward::Function # Function for calculating gradients
    prev::Set{ScalarValue} # Previous values in the graph
    op::String # Operation applied to the value
end

# Base Functions

# Overloads the + operator for ScalarValue types. 
# If the input is not a ScalarValue, it converts it to one.
# Returns a new ScalarValue with the sum of the inputs, and sets the backward function
# to calculate gradients for both inputs.
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

# Overloads the - operator for ScalarValue types.
# Returns the sum of the input and the negation of the input.
function Base.:(-)(a::ScalarValue, b)
    return a + (-b)
end

# Overloads the * operator for ScalarValue types.
# If the input is not a ScalarValue, it converts it to one.
# Returns a new ScalarValue with the product of the inputs, and sets the backward function
# to calculate gradients for both inputs.
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

# Overloads the ^ operator for ScalarValue types and Real inputs.
# Returns a new ScalarValue with the power of the input and exponent, and sets the backward function
# to calculate gradients for the input.
function Base.:(^)(a::ScalarValue, b::Real)
    output = ScalarValue(a.data ^ b, 0.0, nothing, Set([a]), "^$b")
    function backwardf()
        a.grad += b * a.data ^ (b - 1) * output.grad
    end
    output.backward = backwardf
    return output
end

# Overloads the - operator for ScalarValue types.
# Returns the product of the input and -1.
function Base.:(-)(a::ScalarValue)
    return a * -1
end

# Overloads the / operator for ScalarValue types and Real inputs.
# Returns the product of the input and the reciprocal of the input.
function Base.:/(a::ScalarValue, b::Real)
    return a * b^(-1)
end

# Defines a relu function for ScalarValue types.
# Returns 0 if the input is less than 0, otherwise the input.
# Sets the backward function to calculate gradients for the input.
function relu(a::ScalarValue)
    output = a.data < 0 ? 0 : ScalarValue(a.data, 0.0, nothing, Set([a]), "relu")
    function backwardf()
        a.grad += (output.data > 0) * output.grad
    end
end

# Defines a backward function for ScalarValue types.
# Traverses the graph in reverse topological order,
# and calls the backward function for each value.
function backward(a::ScalarValue)
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
        v.backward()
    end
end
