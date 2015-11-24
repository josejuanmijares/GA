
# push!(LOAD_PATH, pwd()); LOAD_PATH=unique(LOAD_PATH);using MyMultiModule
# workspace(); reload(" MyMultiModule"); using  MyMultiModule;


module MyGAMultiPopulation
																												
	export GAmodels,GAreset,GAselect,GAevaluate,GAcrossover,GAmutate,runga

	type GAmodels
		x::Array{Float32,2}							## input values (M populations of N elements )
		NumOfElements::Int64												## length of x
		NumOfPopulations::Int64												## number of populations
		MaxGenerations::Int64												## max generations
		
		NumOfBins::Int64												## number of bins
		
		q::Array{Float32,2}							## q values
		s::Array{Float32,1}							## s values
		score::Float32									## quality level




		function GAmodels()
			return GAmodels(32,8)
		end

		function GAmodels(N_in::Int64, M_in::Int64)
			this =new()
			this.x = rand(Float32, N_in, M_in)
			this.N = N_in
			this.M = M_in
			this.score = 0.0f0
			return this
		end

		function GAmodels(x_in::Array{Float32,2})
			this = new()
			this.x = x_in
			this.N = size(x_in)[1]
			this.M = size(x_in)[2]
			this.score = 0.0f0
			return this
		end
		
		function GAmodels(x_in::Array{Float64,2})
			return GAmodels(convert(Array{Float32,2}, x_in))
		end
		
		
	end
	
	function GAreset(g::GAmodels)
		empty!(g.x);
		g.N = 0;
		g.M = 0;
		g.score = 0.0f0
	end

	function GAselect(g::GAmodels, f::Function, vargs...)
		f(g.x,vargs)
	end

	function GAevaluate(g::GAmodels, f::Function,vargs...)
		g.score = f(g.x,vargs)
	end

	function GAcrossover(g::GAmodels, f::Function,vargs...)
		f(g.x,vargs)
	end
	
	function GAmutate(g::GAmodels, f::Function,vargs...)
		f(g.x,vargs)
	end

	function runga(g::GAmodels, target::Float32, select_func::Function, eval_func::Function, crossover_func::Function, mutation_func::Function)
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