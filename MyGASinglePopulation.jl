
# push!(LOAD_PATH, pwd()); LOAD_PATH=unique(LOAD_PATH); using MyModule
# workspace(); reload("MyModule"); using MyModule;


module MyGASinglePopulation

	export GAmodel,GAreset,GAselect,GAevaluate,GAcrossover,GAmutate,runga

	type GAmodel
		x::Array{Float32,1}							## input values
		N::Int64												## length of x
		#M::Int64												## number of populations
		score::Float32									## quality level

		function GAmodel()
			return GAmodel(32)
		end

		function GAmodel(N_in::Int64)
			this =new()
			this.x = rand(Float32, N_in)
			this.N = N_in
			#this.M = M_in
			this.score = 0.0
			return this
		end

		function GAmodel(x_in::Array{Float64,1})
			this =new()
			this.x = convert(Array{Float32,1}, x_in)
			this.N = length(this.x)
			#this.M = size(this.x)[2]
			this.score = 0.0
			return this
		end

	end

	function GAreset(g::GAmodel)
		empty!(g.x);
		g.N = 0;
		g.score = Float32(0.0);
	end

	function GAselect(g::GAmodel, f::Function,vargs...)
		f(g.x,vargs)
	end
	
	function GAevaluate(g::GAmodel, f::Function,vargs...)
		g.score = f(g.x,vargs)
	end

	function GAcrossover(g::GAmodel, f::Function,vargs...)
		f(g.x,vargs)
	end
	
	function GAmutate(g::GAmodel, f::Function,vargs...)
		f(g.x,vargs)
	end

	function runga(g::GAmodel, target::Float32, select_func::Function, eval_func::Function, crossover_func::Function, mutation_func::Function)
		score_hist =[]
		GAevaluate(g,eval_func)
		while g.score > target
			GAselect(g,select_func)
			GAcrossover(g,crossover_func)
			GAmutate(g,mutation_func)
			GAevaluate(g,eval_func)
			score_hist=[score_hist; g_score]
		end
		
		return g.x, score_hist
	end

	println("MyModule v0.1 is loaded")
	
end




# type GAmodel
#     initial_pop_size::Int
#     gen_num::Int
#
#     population::Array
#
#     rng::AbstractRNG
#
#     GAmodel() = new(0, 1, Any[], EntityData[], EntityData[], MersenneTwister(time_ns()), nothing)
# end