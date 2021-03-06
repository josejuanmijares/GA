# push!(LOAD_PATH, pwd()); LOAD_PATH=unique(LOAD_PATH);using DummyTest
# workspace(); reload("DummyTest"); using  DummyTest;

module DummyTest

	export population, GAmodels, GAmodel

	import ExactHistEq.EHEfast3
	import Goertzel_functions.goertzel, Goertzel_functions.online_variance

	type population
		x													::Array{Float32,1}							# population elements
		N													::Int64													# length of the population
		score											::Float32												# fitness
		q1												::Array{Float32,1}							# q1 values
		q2												::Array{Float32,1}							# q2 values
		M1												::Float32												# mean value
		M2												::Float32												# variance*(N-1) value
		NSamples									::Int64													# total number of samples
		Nbins											::Int64
		
		repopulate								::Function
		
		function population(N::Int64, Nbins::Int64)
			this = new()
			this.Nbins = Nbins
			this.x = EHEfast3(rand(Float32,N),Nbins,Float32)
			this.N = N
			this.score = 0.0f0
			this.q1 = zeros(Float32,this.N+1)
			this.q2 = zeros(Float32,this.N+1)
			this.M1 =0.0f0
			this.M2 =0.0f0
			this.NSamples = 0
			
			this.repopulate = function( )
				this.x = EHEfast3(rand(Float32,this.N), this.Nbins, Float32)
			end
			
			return this
		end
		# function population(x::Array{Float32,1})
		# 	this = new()
		# 	this.x = x
		# 	this.N = length(x)
		#
		# 	this.score = 0.0f0
		# 	this.q1 = zeros(Float32,this.N+1)
		# 	this.q2 = zeros(Float32,this.N+1)
		# 	this.M1 =0.0f0
		# 	this.M2 =0.0f0
		# 	this.NSamples = 0
		# 	return this
		# end
		function population()
			return population(128,16)
		end
	end

	type GAmodels
		ga_pops												::Array{population,1}						# array of populations
		N															::Int64													# Number of populations
		q1														::Array{Float32,1}							# q1 values
		q2														::Array{Float32,1}							# q2 values
		NFreqBins											::UInt64												# number of frequency bins for Groetzel FFT
		M1														::Float32												# mean value
		M2														::Float32												# variance*(N-1) value
		NSamples											::Int64													# total number of samples
		S															::Float32												# standard deviation from best population
		id														::Int64													# id of the best population
		
		getBest												::Function
		evaluateAll										::Function
		par_evaluateAll								::Function
		evaluate											::Function
		randomSelection								::Function
		topN_selection								::Function
		orderedCrossOver							::Function
		exchangeMutation							::Function
		printAll											::Function
		updateBest										::Function
		elitistSelection							::Function
		elitistOrderedCrossOver				::Function
		elitistExchangeMutation				::Function
		groupElitistOrderedCrossOver	::Function
		groupElitistExchangeMutation	::Function
		
		function GAmodels(N::Int64, M::Int64, Nbins::Int64, q1::Array{Float32,1}, q2::Array{Float32,1})
			this = new()
			
			this.N = N																							# initialization
			this.ga_pops = [population(M, Nbins) for i=1:N]
			this.NFreqBins = N
			this.q1 = q1 #zeros(Float32,this.NFreqBins+1)
			this.q2 = q2 #zeros(Float32,this.NFreqBins+1)
			this.M1 =0.0f0
			this.M2 =0.0f0
			this.NSamples = 0
			this.id = 0
			
			######################################################### functions			
			this.updateBest = function(ind, score,target_score)
				this.id = ind
				this.S = score
				
				if this.S < target_score
					this.q1 = this.ga_pops[ind].q1
					this.q2 = this.ga_pops[ind].q2
					this.M1 = this.ga_pops[ind].M1
					this.M2 = this.ga_pops[ind].M2
					this.NSamples = this.ga_pops[ind].NSamples
				end
				
			end
			
			this.getBest = function()
				score = this.ga_pops[1].score
				ind = 1;
				for k=2:this.N
					if score > this.ga_pops[k].score
						score = this.ga_pops[k].score
						ind = k;
					end
				end
				
				#this.updateBest(ind,score)
				
				return ind, score
			end
				
			this.evaluateAll = function()
				for k = 1:this.N
					this.evaluate(k)
				end
			end
			
			this.par_evaluateAll = function()
				pmap(this.evaluate,[1:this.N;])
			end
			
#			this.evaluateFFT = function(k::Int64)
# 				data = this.ga_pops[k].x
# 				psd = real(fft(data).*conj(fft(data)))/length(data)
#
# 			end


			function evaluate(data::Array{Float32,1})
				psd = zeros(Float32,this.NFreqBins+1)
				q1 = zeros(Float32,this.NFreqBins+1)
				q2 = zeros(Float32,this.NFreqBins+1)
				M1 = 0
				M2 = 0
				Nout = 0

				psd, q1, q2 = goertzel(data, this.NFreqBins, this.q1, this.q2)
				M1 = mean(psd)
				M2 = var(psd)
				s = std(psd)
				
				return s
			end
			
			function evaluate(k::Int64)
				psd = zeros(Float32,this.NFreqBins+1)
				q1 = zeros(Float32,this.NFreqBins+1)
				q2 = zeros(Float32,this.NFreqBins+1)
				M1 = 0
				M2 = 0
				Nout = 0
				data = this.ga_pops[k].x
				
				psd, q1, q2 = goertzel(data, this.NFreqBins, this.q1, this.q2)
				M1 = mean(psd)
				M2 = var(psd)
				s = std(psd)
				#s, Nout, M1, M2= online_variance(psd, this.NSamples, this.M1, this.M2)
				
				this.ga_pops[k].q1 = q1
				this.ga_pops[k].q2 = q2
				this.ga_pops[k].M1 = M1
				this.ga_pops[k].M2 = M2
				this.ga_pops[k].q1 = q1
				this.ga_pops[k].NSamples = Nout
				this.ga_pops[k].score = s
			end
			
			this.evaluate = evaluate
			
			this.elitistSelection = function()
				this.evaluateAll()
				k_prime, _ = this.getBest()
				#list = symdiff(k_prime,[1:this.N;])
				list = [1:this.N;]
				#parent1_indexes = [k_prime for k=1:(this.N-1)]
				parent1_indexes = [k_prime for k=1:(this.N)]
				parent2_indexes = list
				
				return convert(Array{Int64,1},parent1_indexes),convert(Array{Int64,1},parent2_indexes)
				
			end
			
			this.randomSelection = function(Npairs::Int64)
				parent1_indexes = zeros(Npairs)
				parent2_indexes = zeros(Npairs)

				list = [1:this.N;]

				parent1_indexes[1] = rand(list)
				list = symdiff(list, parent1_indexes)
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
			
			this.elitistOrderedCrossOver = function(parent1_indexes::Array{Int64,1},parent2_indexes::Array{Int64,1})
				Npairs = length(parent1_indexes)
				
				for k=1:Npairs
					pop_ind1 = parent1_indexes[k]
					pop_ind2 = parent2_indexes[k]
					if pop_ind1 != pop_ind2
						p1 = this.ga_pops[ pop_ind1 ].x 
						p2 = this.ga_pops[ pop_ind2 ].x

						ind1 = sortperm(p1)
						ind2 = sortperm(p2)

						cut_A = rand([1:(length(p1)-1);])
						cut_B = rand([(cut_A+1):length(p1);])

						#child1 = zeros(Float32,size(p1))
						child2 = zeros(Float32,size(p1))
					
						#mask1 = ones(Int64,size(p1))
						mask2 = ones(Int64,size(p1))
					
						#child1[cut_A:cut_B] = p1[cut_A:cut_B]
						child2[cut_A:cut_B] = p2[cut_A:cut_B]
					
						#temp1 = []
						temp2 = []
						for k0=1:length(mask2)
							#if (ind1[k0]>= cut_A) && (ind1[k0]<=cut_B)
							#	mask1[ind2[k0]]= 0
							#end
							if (ind2[k0]>= cut_A) && (ind2[k0]<=cut_B)
								mask2[ind1[k0]]= 0
							end
						end
						for k0 = 1:length(mask2)
							#if mask1[k0]==1
							#	temp1 = [temp1; p2[k0]]
							#end
							if mask2[k0]==1
								temp2 = [temp2; p1[k0]]
							end
						end
					
						k_ind = 1 
						for k0 = 1: (cut_A-1)
							#child1[k0] = temp1[k_ind]
							child2[k0] = temp2[k_ind]
							k_ind +=1
						end
						for k0 = (cut_B+1):length(p1)
							#child1[k0] = temp1[k_ind]
							child2[k0] = temp2[k_ind]
							k_ind +=1
						end

						#this.ga_pops[ pop_ind1 ].x = child1
						this.ga_pops[ pop_ind2 ].x = child2
					end
				end
				
			end
			
			this.groupElitistOrderedCrossOver = function(parent1_indexes::Array{Int64,1},parent2_indexes::Array{Int64,1})
				Npairs = length(parent1_indexes)
				
				for k=1:Npairs
					pop_ind1 = parent1_indexes[k]
					pop_ind2 = parent2_indexes[k]
					if pop_ind1 != pop_ind2
						p1 = this.ga_pops[ pop_ind1 ].x 
						p2 = this.ga_pops[ pop_ind2 ].x

						ind1 = sortperm(p1)
						ind2 = sortperm(p2)

						cut_A = rand([1:(length(p1)-1);])
						cut_B = rand([(cut_A+1):length(p1);])

						child1 = zeros(Float32,size(p1))
						child2 = zeros(Float32,size(p1))
					
						mask1 = ones(Int64,size(p1))
						mask2 = ones(Int64,size(p1))
					
						child1[cut_A:cut_B] = p1[cut_A:cut_B]
						child2[cut_A:cut_B] = p2[cut_A:cut_B]
					
						temp1 = []
						temp2 = []
						for k0=1:length(mask2)
							if (ind1[k0]>= cut_A) && (ind1[k0]<=cut_B)
								mask1[ind2[k0]]= 0
							end
							if (ind2[k0]>= cut_A) && (ind2[k0]<=cut_B)
								mask2[ind1[k0]]= 0
							end
						end
						for k0 = 1:length(mask2)
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

						# replace only if they are better than the parents
						if this.evaluate(p1) > this.evaluate(child1)
							this.ga_pops[ pop_ind1 ].x = child1
						end
						
						if this.evaluate(p2) > this.evaluate(child2)
							this.ga_pops[ pop_ind2 ].x = child2
						end
						
					end
				end
					
					
			end
			
			this.orderedCrossOver = function(parent1_indexes::Array{Int64,1},parent2_indexes::Array{Int64,1})
				Npairs = length(parent1_indexes)
				
				for k=1:Npairs
					pop_ind1 = parent1_indexes[k]
					pop_ind2 = parent2_indexes[k]
					if pop_ind1 != pop_ind2
						p1 = this.ga_pops[ pop_ind1 ].x 
						p2 = this.ga_pops[ pop_ind2 ].x

						ind1 = sortperm(p1)
						ind2 = sortperm(p2)

						cut_A = rand([1:(length(p1)-1);])
						cut_B = rand([(cut_A+1):length(p1);])

						child1 = zeros(Float32,size(p1))
						child2 = zeros(Float32,size(p1))
					
						mask1 = ones(Int64,size(p1))
						mask2 = ones(Int64,size(p1))
					
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

						this.ga_pops[ pop_ind1 ].x = child1
						this.ga_pops[ pop_ind2 ].x = child2
					end
				end
				
			end

			this.elitistExchangeMutation = function(parent1_indexes::Array{Int64,1},parent2_indexes::Array{Int64,1}, prob_mutation::Float64)
				Npairs = length(parent1_indexes)
				for k=1:Npairs
					if rand() < prob_mutation
						pop_ind1 = parent1_indexes[k]
						pop_ind2 = parent2_indexes[k]
						if pop_ind1!=pop_ind2
							#p1 = this.ga_pops[ pop_ind1 ].x 
							p2 = this.ga_pops[ pop_ind2 ].x
						
							#A = rand([1:(length(p1)-1);])
							#B = rand([(A+1):length(p1);])
							#temp = p1[A]
							#p1[A] = p1[B]
							#p1[B] =temp
						
							A = rand([1:(length(p2)-1);])
							B = rand([(A+1):length(p2);])
							temp = p2[A]
							p2[A] = p2[B]
							p2[B] =temp
						
							#this.ga_pops[ pop_ind1 ].x = p1
							this.ga_pops[ pop_ind2 ].x = p2
						end
					end
				end
			end
			
			this.groupElitistExchangeMutation = function(parent1_indexes::Array{Int64,1},parent2_indexes::Array{Int64,1}, prob_mutation::Float64)
				Npairs = length(parent1_indexes)
				for k=1:Npairs
					pop_ind1 = parent1_indexes[k]
					pop_ind2 = parent2_indexes[k]
					if pop_ind1!=pop_ind2
						p1 = this.ga_pops[ pop_ind1 ].x 
						p2 = this.ga_pops[ pop_ind2 ].x
					
						child1 = copy(p1)
						child2 = copy(p2)
					
						c1_ind1 = []
						c1_ind2 = []
						c1_ind1 = rand([1:length(p1);])
						c1_ind2 = rand(setdiff([1:length(p1);], c1_ind1))
						
						c2_ind1 = []
						c2_ind2 = []
						c2_ind1 = rand([1:length(p1);])
						c2_ind2 = rand(setdiff([1:length(p1);], c2_ind1))
						
						for l=2:(length(p1)/2)
							c1_ind1 = [c1_ind1; rand(setdiff([1:length(p1);], union(c1_ind1, c1_ind2))) ]
							c1_ind2 = [c1_ind2; rand(setdiff([1:length(p1);], union(c1_ind1, c1_ind2))) ]
							c2_ind1 = [c2_ind1; rand(setdiff([1:length(p1);], union(c2_ind1, c2_ind2))) ]
							c2_ind2 = [c2_ind2; rand(setdiff([1:length(p1);], union(c2_ind1, c2_ind2))) ]
						end
					
						for l=1:length(c1_ind1)
							if rand() < prob_mutation
								temp = child1[c1_ind1[l]]
								child1[c1_ind1[l]] = child1[c1_ind2[l]]
								child1[c1_ind2[l]]= temp
							end
							if rand() < prob_mutation
								temp = child2[c2_ind1[l]]
								child2[c2_ind1[l]] = child2[c2_ind2[l]]
								child2[c2_ind2[l]]= temp
							end
						end
					
						# replace only if they are better than the parents
						
						#println("eval p1=$(this.evaluate(p1)) > $(this.evaluate(child1)) =child1 ?")
						if this.evaluate(p1) > this.evaluate(child1)
							this.ga_pops[ pop_ind1 ].x = child1
						end
						#println("eval p2=$(this.evaluate(p2)) > $(this.evaluate(child2)) =child2 ?")
						if this.evaluate(p2) > this.evaluate(child2)
							this.ga_pops[ pop_ind2 ].x = child2
						end
					end
				end
			end
			
			this.exchangeMutation = function(parent1_indexes::Array{Int64,1},parent2_indexes::Array{Int64,1}, prob_mutation::Float64)
				Npairs = length(parent1_indexes)
				for k=1:Npairs
					if rand() < prob_mutation
						pop_ind1 = parent1_indexes[k]
						pop_ind2 = parent2_indexes[k]
						p1 = this.ga_pops[ pop_ind1 ].x 
						p2 = this.ga_pops[ pop_ind2 ].x
						
						A = rand([1:(length(p1)-1);])
						B = rand([(A+1):length(p1);])
						temp = p1[A]
						p1[A] = p1[B]
						p1[B] =temp
						
						A = rand([1:(length(p1)-1);])
						B = rand([(A+1):length(p1);])
						temp = p2[A]
						p2[A] = p2[B]
						p2[B] =temp
						
						this.ga_pops[ pop_ind1 ].x = p1
						this.ga_pops[ pop_ind2 ].x = p2
					end
				end
			end
			
			this.printAll = function()

				println("#################################################################################################################################################")
				println(" ::Int64               # Number of populations       |          N = $(this.N)")
				println(" ::Array{Float32,1}    # q1 values (best)            |         q1 = $(this.q1)")
				println(" ::Array{Float32,1}    # q2 values (best)            |         q2 = $(this.q2)")
				println(" ::UInt64              # Groetzel FFT nfreq bins     |  NFreqBins = $(this.NFreqBins)")
				println(" ::Float32             # mean value                  |         M1 = $(this.M1)")
				println(" ::Float32             # variance*(N-1) value        |         M2 = $(this.M2)")
				println(" ::Int64               # total number of samples     |   NSamples = $(this.NSamples)")
				println(" ::Float32             # st dev (best) population    |          S = $(this.S)")
				println(" ::Int64               # total number of samples     |   NSamples = $(this.NSamples)")
				println("#################################################################################################################################################")
				println(" ")
				println("#################################################################################################################################################")
				
				for k=1:this.N
					println(" ::Int64               # population ID               |          k = $(k)")
					println(" ::Array{Float32,1}    # poulation elements          |          x = $(this.ga_pops[k].x)")
					println(" ::Int64               # length of the population    |          N = $(this.ga_pops[k].N)")
					println(" ::Float32             # population fitness value    |      score = $(this.ga_pops[k].score)")
					println(" ::Array{Float32,1}    # q1 values (population)      |         q1 = $(this.ga_pops[k].q1)")
					println(" ::Array{Float32,1}    # q2 values (population)      |         q2 = $(this.ga_pops[k].q2)")
					println(" ::Float32             # mean value (pop.)           |         M1 = $(this.ga_pops[k].M1)")
					println(" ::Float32             # variance*(N-1) value(pop)   |         M2 = $(this.ga_pops[k].M2)")
					println(" ::Int64               # total number of samples     |   NSamples = $(this.ga_pops[k].NSamples)")
					println("-----------------------------------------------------------------------------------------------------------------------------------------------")
				end

			
			end
			
			return this
		end
		
		function GAmodels(N::Int64, M::Int64, Nbins::Int64)
			return GAmodels(N,M,Nbins, zeros(Float32,N+1),zeros(Float32,N+1))
		end
		
		function GAmodels(N::Int64)
			return GAmodels(N,128,16, zeros(Float32,N+1),zeros(Float32,N+1))
		end
		
		function GAmodels()
			return GAmodels(16,128,16,zeros(Float32,16+1),zeros(Float32,16+1))
		end
		
	end	
	
	
	type GAmodel
		ga_pops												::Array{population,1}						# array of populations
		N															::Int64													# Number of populations
		NSamples											::Int64													# total number of samples
		S															::Float32												# standard deviation from best population
		bestId												::Int64													# id of the best population

		evaluate											::Function
		evaluateAll										::Function
		repopulateAll									::Function
		getBest												::Function
		
		rouletteWheelSelection				::Function
		rankSelection									::Function
		tournamentSelection						::Function
		
		orderOneCrossOver							::Function
		orderedCrossOver 							::Function
		partialMatchCrossOver					::Function
		cycleCrossOver								::Function
		
		scrambleMutation							::Function
		inversionMutation							::Function
		swapMutation									::Function
		exchangeMutation							::Function
		insertMutation								::Function
		
		elitistSelection							::Function
		elitistOrderedCrossOver				::Function

		function GAmodel(N::Int64, M::Int64, Nbins::Int64, q1::Array{Float32,1}, q2::Array{Float32,1})
			this = new()
																																# initialization
			this.ga_pops = [population(M, Nbins) for i=1:N]
			this.N = N
			this.NSamples = 0

			######################################################### functions
		
			function evaluate(data::Array{Float32,1})
				N = length(data)
				No2 = Int64(N/2)
				psd = zeros(Float32,No2)
				psd = ((abs( fft(data) )./No2 )[1:No2]).^2;
				s = std(psd)
				return s
			end
			function evaluate(ind::Int64)
				N = length(this.ga_pops[ind].x)
				No2 = Int64(N/2)
				psd = zeros(Float32,No2)
				data = this.ga_pops[ind].x
				psd = ((abs( fft(data) )./No2 )[1:No2]).^2;
				s = std(psd)
				this.ga_pops[ind].score = s
				return s
			end
			this.evaluate = evaluate
			this.evaluateAll = function()
				for k=1:this.N
					this.ga_pops[k].score= this.evaluate(k)
				end
			end	
			this.repopulateAll = function(exceptID::Int64)
				for k=1:(this.N)
					if k!=exceptID
						this.ga_pops[k].repopulate()
					end
				end
			end
			function getBest(dataA::Array{Float32,1}, dataB::Array{Float32,1})
				if this.evaluate(dataA) > this.evaluate(dataB)
					return dataB
				else
					return dataA
				end
			end
			function getBest(A::Int64, B::Int64)
				dataA = this.ga_pops[A].x
				dataB = this.ga_pops[B].x
				ScoreA = this.evaluate(dataA)
				ScoreB = this.evaluate(dataB)
 				if ScoreA > ScoreB
					this.bestId = B
					this.S = ScoreB
					return B, ScoreB,dataB
				else
					this.bestId = A
					this.S = ScoreA
					return A, ScoreA,dataA
				end	
			end
			function getBest()
				b_score = this.ga_pops[1].score
				b_ind = 1
				for k=2:this.N
					if this.ga_pops[k].score < b_score
						b_score = this.ga_pops[k].score
						b_ind = k
					end
				end
				return b_ind, b_score, this.ga_pops[b_ind].x
			end
			this.getBest = getBest


			function rouletteWheelSelection()
				this.evaluateAll()
				S = [this.ga_pops[k].score for k=1:this.N]
				S = cumsum(S./sum(S))
				pA_ids = [findfirst(S.>=rand()) for k=1:this.N]
				pB_ids = [findfirst(S.>=rand()) for k=1:this.N]
				return pA_ids, pB_ids
			end
			function rankSelection()
				this.evaluateAll()
				S = [this.ga_pops[k].score for k=1:this.N]
				Sind = sortperm(S)
				pA_ids =[ Sind[rand([1:this.N])] for k=1: this.N]
				pB_ids =[ Sind[rand([1:this.N])] for k=1: this.N]
				return pA_ids, pB_ids
			end
			function tournamentSelection(TournamentSize::Int64)
				pA_ids[k]= zeros(Int64,this.N)
				pB_ids[k]= zeros(Int64,this.N)
				if TournamentSize<= this.N
					for k=1:this.N
						ind = [rand([1:this.N]) for k=1:TournamentSize]
						S = [this.evaluate(ind[k]) for k=1:TournamentSize]
						pA_ids[k]= findmax(S)[2]
						ind = [rand([1:this.N]) for k=1:TournamentSize]
						S = [this.evaluate(ind[k]) for k=1:TournamentSize]
						pB_ids[k]= findmax(S)[2]
					end
				else
					println("Tournament Size > number of populations")
				end
				return pA_ids, pB_ids
			end
			
			this.rouletteWheelSelection = rouletteWheelSelection
			this.rankSelection = rankSelection
			this.tournamentSelection = tournamentSelection
			
			function order1XO(ind1::Array{Int64,1},ind2::Array{Int64,1}, p1::Array{Float32,1}, p2::Array{Float32,1})
				ini_Pt = rand([1:(length(ind1) - 1 );]);
				end_Pt = rand([(ini_Pt + 1):(length(ind1));]);
				child1 = zeros(Float32,length(p1))
				child1_ind = zeros(Int64, length(p1))
				p1_ind = [find(p1[k].== p1[ind1])[1] for k=1:length(p1)];
				p2_ind = [find(p2[k].== p2[ind2])[1] for k=1:length(p2)];
				child1_ind[ini_Pt:end_Pt] = p1_ind[ini_Pt:end_Pt]
				child1[ini_Pt:end_Pt] = p1[ini_Pt:end_Pt]
				lind = end_Pt+1
				for k=(end_Pt+1):length(ind1)
					if any(p2_ind[k].== child1_ind) == false
						if lind == (length(ind1)+1)
							lind = 1
						end
						child1_ind[lind] = p2_ind[k]
						child1[lind] = p2[ p2_ind[k] ]
						lind += 1
					end
				end	
				for k=1:end_Pt
					if any(p2_ind[k].== child1_ind) == false
						if lind == (length(ind1)+1)
							lind = 1
						end
						child1_ind[lind] = p2_ind[k]
						child1[lind] = p2[ p2_ind[k] ]
						lind += 1
					end
				end	
				return child1
			end
			function orderOneCrossOver(pA_ids::Array{Int64,1},pB_ids::Array{Int64,1})
				child1 = GAmodel(this.N,this.ga_pops[1].N,this.ga_pops[1].Nbins)
				for k=1:this.N
					p1_ind = pA_ids[k]
					p2_ind = pB_ids[k]
					p1 = this.ga_pops[p1_ind].x
					p2 = this.ga_pops[p2_ind].x
					ind1 = sortperm(p1)
					ind2 = sortperm(p2)
					child1.ga_pops[p1_ind].x = order1XO(ind1,ind2, p1,p2)
					child1.ga_pops[p2_ind].x = order1XO(ind2,ind1, p2,p1)
				end
				return child1
			end
			function PMX(ind1::Array{Int64,1},ind2::Array{Int64,1}, p1::Array{Float32,1}, p2::Array{Float32,1})
				ini_Pt = rand([1:(length(ind1) - 1 );]);
				end_Pt = rand([(ini_Pt + 1):(length(ind1));]);
				child1 = zeros(length(p1))
				child1[ini_Pt:end_Pt] = p1[ini_Pt:end_Pt]
				p1_ind = [find(p1[k].== p1[ind1])[1] for k=1:length(p1)];
				p2_ind = [find(p2[k].== p2[ind2])[1] for k=1:length(p2)];
				mask1 = zeros(Int64,length(p1));
				mask1[ini_Pt:end_Pt] = 1;
				child1_ind = zeros(Int64,length(p1));
				child1_ind[ini_Pt:end_Pt]= p1_ind[ini_Pt:end_Pt];
										
				for k = ini_Pt:end_Pt
					if any( p2_ind[k].==child1_ind)==false
						i = p2_ind[k]
						j = p1_ind[k]
						rj = find(p2_ind.==j)
						if mask1[rj][1]==0
							child1[rj] = p2[i]
							child1_ind[rj] = i
							mask1[rj] = 1
						else
							newk = find(p2_ind.==j)
							jj = p1_ind[newk]
							rjj = find(p2_ind.==jj)
							child1[rjj] = p2[i]
							child1_ind[rjj] = i
							mask1[rjj] = 1
						end
					end
				end
				while( any(mask1.==0) ==true )
					k = findfirst(mask1.==0)
					child1[k] = p2[k]
					child1_ind[k] = p2_ind[k]
					mask1[k] = 1
				end
				return child1
			end
			function partialMatchCrossOver(parent1_indexes::Int64,parent2_indexes::Int64)
				partialMatchCrossOver([parent1_indexes],[parent2_indexes])
			end
			function partialMatchCrossOver(parent1_indexes::Array{Int64,1},parent2_indexes::Array{Int64,1})
				child = GAmodel(this.N,this.ga_pops[1].N,this.ga_pops[1].Nbins)
				for k=1:length(parent1_indexes)
					p1_ind = parent1_indexes[k]
					p2_ind = parent2_indexes[k]
					p1 = this.ga_pops[p1_ind].x
					p2 = this.ga_pops[p2_ind].x
					
					ind1 = sortperm(p1)
					ind2 = sortperm(p2)
					child1 = PMX(ind1,ind2,p1,p2)
					child2 = PMX(ind2,ind1,p2,p1)
					
					child.ga_pops[p1_ind].x = child1
					child.ga_pops[p2_ind].x = child2
				end
				return child
			end	

			function cycleXO(ind1::Array{Int64,1},ind2::Array{Int64,1}, p1::Array{Float32,1}, p2::Array{Float32,1})
				k_start = 1
				k_ind = 1
				seq1 = []
				seq1 = convert(Array{Int64,1},seq1)
				seq1_val = []
				
				p1_ind = [find(p1[k].== p1[ind1])[1] for k=1:length(p1)];
				p2_ind = [find(p2[k].== p2[ind2])[1] for k=1:length(p2)];
				
				#println( "p1_ind = $(p1_ind)" )
				#println( "p2_ind = $(p2_ind)" )
				
				child1 = zeros(Float32,length(p1))
				child2 = zeros(Float32,length(p1))
				
				#form the sequence seq1
				pos = 1
				pos_ini = 1
				
				while length(seq1) <length(p1_ind)
					while true
						val_ind = find(p2_ind[pos] .== p1_ind)[1]
						seq1 = [seq1, val_ind]
						child1[val_ind] =  p1[p1_ind[val_ind]]   #p1[ p1_ind[val_ind] ]
						child2[val_ind] =  p2[p2_ind[val_ind]]
						#println( "val_ind = $(val_ind) \t p2_ind[pos] = $(p2_ind[pos]) \t pos=$(pos)" )
						#println( "seq1 = $(seq1)" )
						#println( "child1 = $(child1) \t child2 = $(child2)" )
						pos = val_ind
						if pos == pos_ini 
							break
						end
					end
					remseq = setdiff(p1_ind, seq1)
					if length(seq1) <length(p1_ind)
						pos = remseq[1]
						pos_ini = remseq[1]
					else
						break
					end
					#println("pos = $pos ")
					while true
						val_ind = find(p1_ind[pos] .== p2_ind)[1]
						seq1 = [seq1, val_ind]
						child1[val_ind] = p2[p2_ind[val_ind]]
						child2[val_ind] = p1[p1_ind[val_ind]] 
						# println( "val_ind = $(val_ind) \t p2_ind[pos] = $(p2_ind[pos]) \t pos=$(pos)" )
						# println( "seq1 = $(seq1)" )
						# println( "child1 = $(child1) \t child2 = $(child2)" )
						pos = val_ind
						if pos == pos_ini 
							break
						end
					end
					remseq = setdiff(p1_ind, seq1)
					if length(seq1) <length(p1_ind)
						pos = remseq[1]
						pos_ini = remseq[1]
					else
						break
					end
					# println("pos = $pos ")
				end
				return child1, child2
			end

			function cycleCrossOver(pA_ids::Array{Int64,1},pB_ids::Array{Int64,1})
				child1 = GAmodel(this.N,this.ga_pops[1].N,this.ga_pops[1].Nbins)
				for k=1:this.N
					p1_ind = pA_ids[k]
					p2_ind = pB_ids[k]
					p1 = this.ga_pops[p1_ind].x
					p2 = this.ga_pops[p2_ind].x
					ind1 = sortperm(p1)
					ind2 = sortperm(p2)
					child1.ga_pops[p1_ind].x, child1.ga_pops[p2_ind].x = cycleXO(ind1,ind2, p1,p2)
				end
				return child1
			end

			this.orderedCrossOver = function(parent1_indexes::Int64,parent2_indexes::Int64)
				pop_ind1 = parent1_indexes
				pop_ind2 = parent2_indexes
				p1 = this.ga_pops[ pop_ind1 ].x 
				p2 = this.ga_pops[ pop_ind2 ].x
				child1 = copy(p1)
				child2 = copy(p2)
				
				if pop_ind1 != pop_ind2
					ind1 = sortperm(p1)
					ind2 = sortperm(p2)

					cut_A = rand([1:(length(p1)-1);])
					cut_B = rand([(cut_A+1):length(p1);])

					#child1 = zeros(Float32,size(p1))
					#child2 = zeros(Float32,size(p1))
				
					mask1 = ones(Int64,size(p1))
					mask2 = ones(Int64,size(p1))
				
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

					#this.ga_pops[ pop_ind1 ].x = child1
					#this.ga_pops[ pop_ind2 ].x = child2
				end
				
				return child1, child2
			end
			this.cycleCrossOver = cycleCrossOver
			this.orderOneCrossOver= orderOneCrossOver
			this.partialMatchCrossOver = partialMatchCrossOver
			
			
			
			function insertMutation(prob_mutation::Float64)
				child1 = GAmodel(this.N,this.ga_pops[1].N,this.ga_pops[1].Nbins)
				for k=1:this.N
					p1 = this.ga_pops[k].x
					if rand()<= prob_mutation
						pos1 = rand([1:(length(p1) - 2);])
						pos2 = rand([pos1+2:length(p1);])
						#println("p1 = $(p1)	\t pos1 = $(pos1) \t pos2 = $(pos2) ")
						insert!(p1, pos1, splice!(p1,pos2))
						#println("p1 = $(p1)	\t pos1 = $(pos1) \t pos2 = $(pos2) ")
					end
					child1.ga_pops[k].x = p1
				end
				return child1
			end
			
			function swapMutation(prob_mutation::Float64)
				child1 = GAmodel(this.N,this.ga_pops[1].N,this.ga_pops[1].Nbins)
				for k=1:this.N
					p1 = this.ga_pops[k].x
					if rand()<= prob_mutation
						pos1 = rand([1:(length(p1) - 1);])
						pos2 = rand([(pos1+1):length(p1);])
						temp = p1[pos1]
						p1[pos1] = p1[pos2]
						p1[pos2]=temp
					end
					child1.ga_pops[k].x = p1
				end
				return child1
			end
			
			function inversionMutation(prob_mutation::Float64)
				child1 = GAmodel(this.N,this.ga_pops[1].N,this.ga_pops[1].Nbins)
				for k=1:this.N
					p1 = this.ga_pops[k].x
					if rand()<= prob_mutation
						pos1 = rand([1:(length(p1) - 1);])
						pos2 = rand([(pos1+1):length(p1);])
						p1[pos1:1:pos2] = p1[pos2:-1:pos1]
					end
					child1.ga_pops[k].x = p1
				end
				return child1
			end
			
			function scrambleMutation(prob_mutation::Float64)
				child1 = GAmodel(this.N,this.ga_pops[1].N,this.ga_pops[1].Nbins)
				for k=1:this.N
					p1 = this.ga_pops[k].x
					if rand()<= prob_mutation
						pos1 = rand([1:(length(p1) - 1);])
						pos2 = rand([(pos1+1):length(p1);])
						p1[pos1:1:pos2] = shuffle(p1[pos1:1:pos2])
					end
					child1.ga_pops[k].x = p1
				end
				return child1
			end
			
			
			this.scrambleMutation = scrambleMutation
			this.inversionMutation =inversionMutation
			this.swapMutation = swapMutation
			this.insertMutation = insertMutation
			this.exchangeMutation = function(parent1_indexes::Int64,parent2_indexes::Int64, prob_mutation::Float64)
				pop_ind1 = parent1_indexes
				pop_ind2 = parent2_indexes
				p1 = this.ga_pops[ pop_ind1 ].x 
				p2 = this.ga_pops[ pop_ind2 ].x

				if rand() < prob_mutation					
					A = rand([1:(length(p1)-1);])
					B = rand([(A+1):length(p1);])
					temp = p1[A]
					p1[A] = p1[B]
					p1[B] =temp
					this.ga_pops[ pop_ind1 ].x = p1
				end
				
				if rand() < prob_mutation	
					A = rand([1:(length(p1)-1);])
					B = rand([(A+1):length(p1);])
					temp = p2[A]
					p2[A] = p2[B]
					p2[B] =temp
					this.ga_pops[ pop_ind2 ].x = p2
				end
					
				return pop_ind1,pop_ind2, p1, p2
			end
			this.elitistSelection = function()
				this.evaluateAll()
				k_prime, _, _ = this.getBest()
				#list = symdiff(k_prime,[1:this.N;])
				list = [1:this.N;]
				#parent1_indexes = [k_prime for k=1:(this.N-1)]
				parent1_indexes = [k_prime for k=1:(this.N)]
				parent2_indexes = list
				return convert(Array{Int64,1},parent1_indexes),convert(Array{Int64,1},parent2_indexes)
			end
			this.elitistOrderedCrossOver = function(parent1_indexes::Array{Int64,1},parent2_indexes::Array{Int64,1})
				for k=1:length(parent1_indexes)
					pop_ind1 = parent1_indexes[k]
					pop_ind2 = parent2_indexes[k]
					
					p1 = copy(this.ga_pops[ pop_ind1 ].x) 
					p2 = copy(this.ga_pops[ pop_ind2 ].x)

					if pop_ind1 != pop_ind2
						ind1 = sortperm(p1)
						ind2 = sortperm(p2)

						cut_A = rand([1:(length(p1)-1);])
						cut_B = rand([(cut_A+1):length(p1);])

						child1 = zeros(Float32,size(p1))
						child2 = zeros(Float32,size(p1))
				
						mask1 = ones(Int64,size(p1))
						mask2 = ones(Int64,size(p1))
				
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

					
						##println("eval(p1) $(this.evaluate(p1))  vs  eval(child1) $(this.evaluate(child1)) \t \t eval(p2) $(this.evaluate(p2)) \t vs  eval(child2) $(this.evaluate(child2))")
						#this.ga_pops[ pop_ind1 ].x = child1
						#this.ga_pops[ pop_ind2 ].x = child2
					end
				end
				#return pop_ind1,pop_ind2, this.ga_pops[ pop_ind1 ].x, this.ga_pops[ pop_ind2 ].x
			end



			
			return this
		end
		
		function GAmodel(N::Int64,M::Int64, Nbins::Int64)
			return GAmodel(N,M,Nbins, zeros(Float32,M+1),zeros(Float32,M+1))
		end
		
		function GAmodel()
			return GAmodel(16,128,16,zeros(Float32,128+1),zeros(Float32,128+1))
		end
		

	
	
		
	end
	

end





