# push!(LOAD_PATH, pwd()); LOAD_PATH=unique(LOAD_PATH);using DummyTest
# workspace(); reload("DummyTest"); using  DummyTest;

module DummyTest

	export population, GAmodels

	import ExactHistEq.EHEfast1
	import Goertzel_functions.goertzel, Goertzel_functions.online_variance

	type population
		x									::Array{Float32,1}											# population elements
		N									::Int64																	# length of the population
		score							::Float32																# fitness
		function population(N::Int64, Nbins::Int64)
			this = new()
			this.x = EHEfast1(rand(Float32,N),Nbins)
			this.N = N
			this.score = 0.0f0
			return this
		end
		function population(x::Array{Float32,1})
			this = new()
			this.x = x
			this.N = length(x)
			this.score = 0.0f0
			return this
		end
		function population()
			return population(128,16)
		end
	end

	type GAmodels
		ga_pops						::Array{population,1}										# array of populations
		N									::Int64																	# Number of populations
		q1								::Array{Float32,1}											# q1 values
		q2								::Array{Float32,1}											# q2 values
		NFreqBins					::UInt64																# number of frequency bins for Groetzel FFT
		M1								::Float32																# mean value
		M2								::Float32																# variance*(N-1) value
		NSamples					::Int64																	# total number of samples
		S									::Float32																# standard deviation from best population
		
		get_best					::Function
		evaluateAll				::Function
		evaluate					::Function
		random_selection	::Function
		topN_selection		::Function
		orderedCrossOver	::Function
		
		function GAmodels(N::Int64, M::Int64, Nbins::Int64)
			this = new()
			
			this.N = N																							# initialization
			this.ga_pops = [population(M, Nbins) for i=1:N]
			this.NFreqBins = 16
			this.q1 = zeros(Float32,this.NFreqBins+1)
			this.q2 = zeros(Float32,this.NFreqBins+1)
			this.M1 =0.0f0
			this.M2 =0.0f0
			this.NSamples = 0
			
			######################################################### functions
			this.get_best = function()
				psd = zeros(Float32,this.NFreqBins+1)
				q1 = zeros(Float32,this.NFreqBins+1)
				q2 = zeros(Float32,this.NFreqBins+1)
				s=Inf; n=0; m1=0.0; m2=0.0;
				
				q1_best = zeros(Float32,this.NFreqBins+1)
				q2_best = zeros(Float32,this.NFreqBins+1)
				s_best=Inf; n_best=0; m1_best=0.0; m2_best=0.0;
				kind = 1
				for k = 1:N
					data = this.ga_pops[k].x
					psd, q1, q2 = goertzel(data, this.NFreqBins, this.q1, this.q2)
					s, n, m1, m2 = online_variance(psd, this.NSamples, this.M1, this.M2)
					this.ga_pops[k].score = s
					if s < s_best
						q1_best = q1
						q2_best = q2
						n_best = n
						m1_best = m1 
						m2_best = m2
						s_best = s
						kind =k
					end
				end
				this.q1 = q1_best; this.q2 = q2_best; this.NSamples = n_best;
				this.M1 = m1_best; this.M2 = m2_best; this.S = s_best;
				return kind
			end
			this.evaluateAll = function()
				psd = zeros(Float32,this.NFreqBins+1)
				for k = 1:N
					data = this.ga_pops[k].x
					psd, _, _ = goertzel(data, this.NFreqBins, this.q1, this.q2)
					s, _, _, _= online_variance(psd, this.NSamples, this.M1, this.M2)
					this.ga_pops[k].score = s
				end
			end
			this.evaluate = function(k::Int64)
				psd = zeros(Float32,this.NFreqBins+1)
				data = this.ga_pops[k].x
				psd, _, _ = goertzel(data, this.NFreqBins, this.q1, this.q2)
				s, _, _, _= online_variance(psd, this.NSamples, this.M1, this.M2)
				this.ga_pops[k].score = s
			end
			this.random_selection = function(Npairs::Int64)
				parent1_indexes = zeros(Npairs)
				parent2_indexes = zeros(Npairs)

				list = [1:this.N;]

				parent1_indexes[1] = rand(list)
				list = setdiff(list,parent1_indexes)
				parent2_indexes[1] = rand(list)
				list = setdiff(list,union(parent1_indexes,parent2_indexes))
				
				for k=2:Npairs
					parent1_indexes[k] = rand(list)
					list = setdiff(list,union(parent1_indexes,parent2_indexes))
					parent2_indexes[k] = rand(list)
					list = setdiff(list,union(parent1_indexes,parent2_indexes))
				end
				
				return convert(Array{Int64,1},parent1_indexes),convert(Array{Int64,1},parent2_indexes)
			end
			
			this.topN_selection = function(Npairs::Int64)
				scores = [this.ga_pops[k].score for k=1:this.N]
				indexes = sortperm(scores)
				parent1_indexes = zeros(Int64,Npairs)
				parent2_indexes = zeros(Int64,Npairs)
				
				for k=1:Npairs
					parent1_indexes[k] = find(indexes.== ((k-1)*2 +1 ) )[1]
					parent2_indexes[k] = find(indexes.== ((k-1)*2 +2 ) )[1]
				end
				return parent1_indexes,parent2_indexes
			end
			
			this.orderedCrossOver = function(parent1_indexes::Array{Int64,1},parent2_indexes::Array{Int64,1})
				Npairs = length(parent1_indexes)
				
				for k=1:Npairs
					pop_ind1 = parent1_indexes[k]
					pop_ind2 = parent2_indexes[k]
					p1 = this.ga_pops[ pop_ind1 ].x 
					#Float32[0.10714543f0,0.24455142f0,0.54020226f0,0.41449094f0,0.5355526f0,0.3036313f0,0.77046824f0,0.7646837f0]
					
					p2 = this.ga_pops[ pop_ind2 ].x
					#Float32[0.27402735f0,0.9973304f0,0.6563084f0,0.2529049f0,0.13000119f0,0.53481257f0,0.0069544315f0,0.8772366f0]
					
					
					ind1 = sortperm(p1)
					ind2 = sortperm(p2)
					
					cut_A = rand([1:(length(p1)-1);])
					#6 #
					cut_B = rand([(cut_A+1):length(p1);])
					#7 #
					child1 = zeros(Float32,size(p1))
					child2 = zeros(Float32,size(p1))
					
					mask1 = ones(Int64,size(p1))
					mask2 = ones(Int64,size(p1))
					
					#println("cut_A = $cut_A cut_B = $cut_B")
					
					#println("p1 = $p1")
					#println("p2 = $p2")
					
					child1[cut_A:cut_B] = p1[cut_A:cut_B]
					child2[cut_A:cut_B] = p2[cut_A:cut_B]
					
					temp1 = []
					temp2 = []
					for k0=1:length(mask1)
						if (ind1[k0]>= cut_A) && (ind1[k0]<=cut_B)
							mask1[ind2[k0]]= 0
						end
						if (ind2[k0]>= cut_A) && (ind2[k0]<=cut_B)
							mask2[ind1[k0]]= 0
						end
					end
					for k0 = 1:length(mask1)
						if mask1[k0]==1
							temp1 = [temp1; p2[k0]]
						end
						if mask2[k0]==1
							temp2 = [temp2; p1[k0]]
						end
					end
					
					k_ind = 1 
					for k0 = 1: (cut_A-1)
						child1[k0] = temp1[k_ind]
						child2[k0] = temp2[k_ind]
						k_ind +=1
					end
					for k0 = (cut_B+1):length(p1)
						child1[k0] = temp1[k_ind]
						child2[k0] = temp2[k_ind]
						k_ind +=1
					end
						
					#println("##### child1 = $child1")
					#println("##### child2 = $child2")
					this.ga_pops[ pop_ind1 ].x = child1
					this.ga_pops[ pop_ind2 ].x = child2
				end
				
			end
			
			
			return this
		end
		
		function GAmodels(N::Int64)
			return GAmodels(N,128,16)
		end
		
		function GAmodels()
			return GAmodels(16,128,16)
		end
		
	end	
	
	function evaluateGA(g::GAmodels)
		## return best population
	end
	

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