module TransformationStrain

using TensorOperations, LinearAlgebra
using FiniteElements, LevelSet, IsotropicElasticity

export assembleUniformMesh


"""
	normal(tangent::Array{Float64, 1})
return a vector orthogonal to `tangent` by taking the cross product of
`tangent` with the z-axis.
"""
function normal(tangent::Array{Float64, 1})
	scale = sqrt(tangent[1]^2 + tangent[2]^2)
	N = [tangent[2], -tangent[1]]/scale
	return N
end

"""
	outer(v1::AbstractArray, v2::AbstractArray)
outer product of vectors `v1` and `v2` in matrix form.
"""
function outer(v1::AbstractArray, v2::AbstractArray)
	K = zeros(length(v1), length(v2))
	for i in 1:length(v1)
		for j in 1:length(v2)
			K[i,j] = v1[i]*v2[j]
		end
	end
	return K
end

"""
	linearForm(FI::Array{Float64, 1}, ∇ϕI::Array{Float64, 1},
		λs::Float64, μs::Float64, θ0::Float64, δ::Array{Float64, 2})
Compute the contraction:
	∇sym(ϕI) * E * ϵt
This is the `[I]` entry of the element right-hand-side
vector. Here `∇sym` refers to the symmetrized gradient.
"""
function linearForm(FI::Array{Float64, 1}, ∇ϕI::Array{Float64, 1},
	λs::Float64, μs::Float64, θ0::Float64, δ::Array{Float64, 2})

	@tensor begin
		FI[p] = 0.5*(δ[i,p]*∇ϕI[j] + δ[j,p]*∇ϕI[i])*((λs*θ0)*δ[i,j] + (2μs*θ0/3)*δ[i,j])
	end
end

"""
	linearForm(mapping::Map{T,dim,spacedim}, assembler::Assembler,
		lambda::Float64, mu::Float64, theta::Array{Float64, 2}) where {T,dim,spacedim}
Assemble the right-hand-side vector representing the forcing function from
isotropic transformation strain `theta`. `lambda` and `mu` are the lame
coefficients assumed constant throughout the element.
"""
function linearForm(mapping::Map{T,dim,spacedim}, assembler::Assembler,
	lambda::Float64, mu::Float64, theta0::Float64) where {T,dim,spacedim}

	kronecker_delta = diagm(0 => ones(spacedim))
	FI = zeros(spacedim)

	Nnodes = length(mapping.master.basis.functions)
	for q in eachindex(mapping.master.quadrature.points)
		(pq, wq) = mapping.master.quadrature[q]
		for I in 1:Nnodes
			∇ϕI = mapping[:gradients][I,q]
			linearForm(FI, ∇ϕI, lambda, mu, theta0, kronecker_delta)
			assembler.element_rhs[I] += FI*mapping[:dx][q]
		end
	end
end

"""
	normalContinuity(assembler::Assembler, scale_by::Float64,
		lineMapping::Map{T1, 1, spacedim}, basis_values::Array{Float64, 2},
		node_ids1::Array{Int64, 1}, node_ids2::Array{Int64, 1}) where {T1, spacedim}
assemble the free slip constraint in `assembler.element_matrix[I,J]`, for `I`
in `node_ids1` and `J` in `node_ids2`, multiplying the constraint by the
`scale_by` factor.
"""
function normalContinuity(assembler::Assembler, scale_by::Float64,
	lineMapping::Map{T1, 1, spacedim}, basis_values::Array{Float64, 2},
	node_ids1::Array{Int64, 1}, node_ids2::Array{Int64, 1}) where {T1, spacedim}

	for q in eachindex(lineMapping.master.quadrature.points)
		n = normal(lineMapping[:jacobian][q][:,1])
		dv = lineMapping[:dx][q]
		outer_product = outer(n, n)
		for I in node_ids1
			NI = basis_values[I,q]
			for J in node_ids2
				NJ = basis_values[J,q]
				assembler.element_matrix[I,J] += 1e6*scale_by*outer_product*dv
			end
		end
	end
end

"""
	normalContinuity(nodes::Array{Float64, 2}, assembler::Assembler,
		lineMapping::Map{T1, 1, spacedim}, surfaceBasis::Basis{T2},
		distance::Array{Float64, 1}) where {T2 <: Triangulation{M,2}} where {T1,M,spacedim}
constrain the normal component of the solution to be continuous on an element
defined by `nodes`. The constraint is added into `assembler`. Continuity is
enforced along the zero level set of `distance`. `lineMapping` is used to
perform 1D quadrature. `surfaceBasis` defines the 2D basis on the element.
"""
function normalContinuity(nodes::Array{Float64, 2}, assembler::Assembler,
	lineMapping::Map{T1, 1, spacedim}, surfaceBasis::Basis{T2},
	distance::Array{Float64, 1}) where {T2 <: Triangulation{M,2}} where {T1,M,spacedim}

	interface_reference_endpoints = interfaceEdgeIntersection(distance,
		surfaceBasis)
	reinit(lineMapping, interface_reference_endpoints)
	basis_values = evaluate(surfaceBasis, lineMapping[:coordinates])
	interface_spatial_endpoints = interpolate(nodes,
		interface_reference_endpoints, surfaceBasis)
	reinit(lineMapping, interface_spatial_endpoints)
	parent_node_ids = findall(x -> x < 0, distance)
	product_node_ids = findall(x -> x > 0, distance)
	normalContinuity(assembler, 1.0, lineMapping, basis_values,
		parent_node_ids, parent_node_ids)
	normalContinuity(assembler, -1.0, lineMapping, basis_values,
		parent_node_ids, product_node_ids)
	normalContinuity(assembler, -1.0, lineMapping, basis_values,
		product_node_ids, parent_node_ids)
	normalContinuity(assembler, 1.0, lineMapping, basis_values,
		product_node_ids, product_node_ids)
end

"""
	assembleUniformMesh(distance::Array{Float64, 1}, mesh::Mesh{spacedim},
    	q_order::Int64, parent_lambda::Float64, parent_mu::Float64,
		product_lambda::Float64, product_mu::Float64, theta0::Float64) where spacedim
Assemble the linear system for transformation strain driven linear elasticity
on a uniform `mesh` object.
"""
function assembleUniformMesh(distance::Array{Float64, 1}, mesh::Mesh{spacedim},
    q_order::Int64, parent_lambda::Float64, parent_mu::Float64,
	product_lambda::Float64, product_mu::Float64, theta0::Float64) where spacedim

    # Check that the mesh has all the necessary information
    @assert length(distance) == size(mesh[:nodes])[2] "nodal distance values must match number of nodes"
    @assert haskey(mesh.data, :element_groups) "Mesh must have element groups"
    @assert haskey(mesh.data[:element_groups], "surface") "Mesh must have an element group for surface elements"
    @assert length(keys(mesh[:element_groups]["surface"])) == 1 "Uniform mesh cannot have multiple element types"

    elType = collect(keys(mesh[:element_groups]["surface"]))[1]
	number_of_nodes = size(mesh[:nodes])[2]

    system_matrix = SystemMatrix()
    system_rhs = SystemRHS()

    parent_assembler = Assembler(elType, spacedim)
    product_assembler = Assembler(elType, spacedim)
    interface_assembler = Assembler(elType, spacedim)

	reinit(parent_assembler)
	reinit(product_assembler)
	reinit(interface_assembler)

    surfaceMapping = Map{elType,spacedim}(q_order, :gradients)
    lineMapping = Map{Line{2},spacedim}(q_order, :coordinates, :gradients)

    # Assemble the linear form using the parent Lame coefficient values
    node_ids = mesh[:elements][elType][:,1]
    nodes = mesh[:nodes][:,node_ids]
	reinit(surfaceMapping, nodes)

    bilinearForm(surfaceMapping, parent_assembler, parent_lambda, parent_mu)
    bilinearForm(surfaceMapping, product_assembler, product_lambda, product_mu)
	linearForm(surfaceMapping, product_assembler, product_lambda, product_mu,
		theta0)

	for elem_id in mesh[:element_groups]["surface"][elType]
        node_ids = mesh[:elements][elType][:, elem_id]
        max_level_set_value = maximum(distance[node_ids])
        min_level_set_value = minimum(distance[node_ids])
        if max_level_set_value*min_level_set_value > 0
            if max_level_set_value < 0
                updateSystemMatrix(system_matrix,
                    parent_assembler.element_matrix, node_ids,
					parent_assembler.ndofs)
            else
                updateSystemMatrix(system_matrix,
                    product_assembler.element_matrix, node_ids,
					product_assembler.ndofs)
				updateSystemRHS(system_rhs,
					product_assembler.element_rhs, node_ids,
					product_assembler.ndofs)
            end
        else
			reinit(interface_assembler)
			nodes = mesh[:nodes][:,node_ids]
			normalContinuity(nodes, interface_assembler, lineMapping,
				surfaceMapping.master.basis, distance[node_ids])
			updateSystemMatrix(system_matrix,
				interface_assembler.element_matrix, node_ids,
				interface_assembler.ndofs)
        end
    end

	system = GlobalSystem(system_matrix, system_rhs, spacedim, number_of_nodes)
	return system
end


# module TransformationStrain ends here
end
# module TransformationStrain ends here
