module TransformationStrain

using FiniteElements, IsotropicElasticity

function assembleSystem(mesh::Mesh{spacedim}, q_order::Int64, lambda::Function,
    mu::Function, θ0::Float64) where spacedim

    # Check that the mesh has all the necessary information
    @assert haskey(mesh.data, :element_groups) "Mesh must have element groups"
    @assert haskey(mesh.data[:element_groups], "parent") "Mesh must have an element group for parent phase"
    @assert haskey(mesh.data[:element_groups], "product") "Mesh must have an element group for product phase"

    system_matrix = SystemMatrix()
    system_rhs = SystemRHS()
    bilinearForm(lambda, mu, "core", mesh, q_order, system_matrix)
end


# module TransformationStrain ends here
end
# module TransformationStrain ends here
