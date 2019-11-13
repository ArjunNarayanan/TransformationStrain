module TransformationStrain

using TensorOperations, LinearAlgebra
using FiniteElements, LevelSet, IsotropicElasticity

export assembleUniformMesh, elementAveragedStrain


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
		outer_product::Array{Float64, 2}, node_ids1::Array{Int64, 1},
		node_ids2::Array{Int64, 1}; penalty = 1e6) where {T1, spacedim}
assemble the free slip constraint in `assembler.element_matrix[I,J]`, for `I`
in `node_ids1` and `J` in `node_ids2`, multiplying the constraint by the
`scale_by` factor.
"""
function normalContinuity(assembler::Assembler, scale_by::Float64,
	outer_product::Array{Float64, 2}, lineMapping, node_ids1::Array{Int64, 1},
	node_ids2::Array{Int64, 1}; penalty = 1e6) where {T1, spacedim}

	for q in eachindex(lineMapping.master.quadrature.points)
		w = lineMapping.master.quadrature.weights[q]
		for I in node_ids1
			for J in node_ids2
				assembler.element_matrix[I,J] += penalty*scale_by*outer_product*w
			end
		end
	end
end

function normalContinuity(assembler::Assembler, outer_product::Array{Float64, 2},
	node_ids::AbstractArray; penalty = 1e6)

	for i in 1:length(node_ids)
		I = node_ids[i]
		for j in (i+1):length(node_ids)
			J = node_ids[j]
			assembler.element_matrix[I,I] += penalty*outer_product
			assembler.element_matrix[J,J] += penalty*outer_product
			assembler.element_matrix[I,J] += -1.0*penalty*outer_product
			assembler.element_matrix[J,I] += -1.0*penalty*outer_product
		end
	end
end

"""
	normalContinuity(nodes::Array{Float64, 2}, assembler::Assembler,
	surfaceBasis::Basis{T2}, distance::Array{Float64, 1}) where {T2 <: Triangulation{M,2}} where {T1,M,spacedim}
constrain the normal component of the solution to be continuous across an implicit
interface. `distance` is the signed distance of each `nodes` to the interface.
The normal to the interface is obtained by a best-fit hyperplane.
The constraint is added into `assembler`.
"""
function normalContinuity(nodes::Array{Float64, 2}, assembler::Assembler,
	surfaceBasis::Basis{T2}, lineMapping,
	distance::Array{Float64, 1}) where {T2 <: Triangulation{M,2}} where {T1,M,spacedim}

	point_on_interface = interfaceEdgeIntersection(nodes, distance, surfaceBasis)
	normal = fitNormal(nodes, distance, point_on_interface)
	outer_product = outer(normal, normal)
	parent_node_ids = findall(x -> x < 0, distance)
	product_node_ids = findall(x -> x >= 0, distance)
	normalContinuity(assembler, 1.0, outer_product, lineMapping, parent_node_ids, parent_node_ids)
	normalContinuity(assembler, -1.0, outer_product, lineMapping, parent_node_ids, product_node_ids)
	normalContinuity(assembler, -1.0, outer_product, lineMapping, product_node_ids, parent_node_ids)
	normalContinuity(assembler, 1.0, outer_product, lineMapping, product_node_ids, product_node_ids)
end

function normalContinuity(nodes::Array{Float64, 2}, assembler::Assembler,
	surfaceBasis::Basis, distance::Array{Float64, 1})

	point_on_interface = interfaceEdgeIntersection(nodes, distance, surfaceBasis)
	normal = fitNormal(nodes, distance, point_on_interface)
	outer_product = outer(normal, normal)
	node_ids = 1:size(nodes)[2]
	normalContinuity(assembler, outer_product, node_ids)
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
		if max_level_set_value == min_level_set_value
			error("Under-resolved interface")
		elseif max_level_set_value*min_level_set_value <= 0 && min_level_set_value < 0
			# Interface element
			reinit(interface_assembler)
			nodes = mesh[:nodes][:,node_ids]
			normalContinuity(nodes, interface_assembler, surfaceMapping.master.basis,
				distance[node_ids])
			updateSystemMatrix(system_matrix,
				interface_assembler.element_matrix, node_ids,
				interface_assembler.ndofs)
		elseif max_level_set_value < 0
			# Parent element
			updateSystemMatrix(system_matrix,
				parent_assembler.element_matrix, node_ids,
				parent_assembler.ndofs)
		else
			# Product element
			updateSystemMatrix(system_matrix,
				product_assembler.element_matrix, node_ids,
				product_assembler.ndofs)
			updateSystemRHS(system_rhs,
				product_assembler.element_rhs, node_ids,
				product_assembler.ndofs)
		end
    end

	system = GlobalSystem(system_matrix, system_rhs, spacedim, number_of_nodes)
	return system
end


"""
	updateStrain(strain::Array{Float64, 2}, elem_id::Int64,
		node_ids::Array{Int64, 1}, nodes::Array{Float64, 2}, surfaceMapping::Map,
		displacement::Array{Float64, 2})
update `strain[:,elem_id]` with the element averaged strain.
"""
function updateStrain(strain::Array{Float64, 2}, elem_id::Int64,
	node_ids::Array{Int64, 1}, nodes::Array{Float64, 2}, surfaceMapping::Map,
	displacement::Array{Float64, 2})

	reinit(surfaceMapping, nodes)
	number_of_nodes = length(surfaceMapping.master.basis.functions)

	for q in eachindex(surfaceMapping.master.quadrature.points)
		w = surfaceMapping.master.quadrature.weights[q]/sum(surfaceMapping.master.quadrature.weights)
		for I in 1:number_of_nodes
			strain[1, elem_id] += surfaceMapping[:gradients][I,q][1]*displacement[1,node_ids[I]]*w
			strain[2, elem_id] += 0.5*( surfaceMapping[:gradients][I,q][2]*displacement[1,node_ids[I]] +
										surfaceMapping[:gradients][I,q][1]*displacement[2,node_ids[I]] )*w
			strain[3, elem_id] += surfaceMapping[:gradients][I,q][2]*displacement[2,node_ids[I]]*w
		end
	end
end

"""
	elementAveragedStrain(mesh::Mesh, displacement::Array{Float64, 2})
compute the element averaged strain for the given `displacement` field on
the elements of `mesh`.
"""
function elementAveragedStrain(mesh::Mesh{spacedim},
	displacement::Array{Float64, 2}; q_order = 2) where spacedim

	@assert spacedim == 2 "Only implemented for 2D problems"
	elType = collect(keys(mesh[:element_groups]["surface"]))[1]
	surfaceMapping = Map{elType,spacedim}(q_order, :gradients)

	number_of_elmts = size(mesh[:elements][elType])[2]
	strain = zeros(3, number_of_elmts)

	for elem_id in 1:number_of_elmts
		node_ids = mesh[:elements][elType][:,elem_id]
		nodes = mesh[:nodes][:,node_ids]
		updateStrain(strain, elem_id, node_ids, nodes, surfaceMapping,
			displacement)
	end
	return strain
end



# module TransformationStrain ends here
end
# module TransformationStrain ends here
