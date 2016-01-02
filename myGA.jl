# push!(LOAD_PATH, pwd()); LOAD_PATH=unique(LOAD_PATH);using DummyTest
# workspace(); reload("DummyTest"); using  DummyTest;

module myGA

	export population,  GAmodel

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
		function population(x::Array{Float32,1})
			this = new()
			this.x = x
			this.N = length(x)

			this.score = 0.0f0
			this.q1 = zeros(Float32,this.N+1)
			this.q2 = zeros(Float32,this.N+1)
			this.M1 =0.0f0
			this.M2 =0.0f0
			this.NSamples = 0
			return this
		end
		function population()
			return population(128,16)
		end
	end

	
	
	type GAmodel
		data												::Array{population,1}						# array of populations
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
		
		positionBasedCrossOver				::Function
		orderOneCrossOver							::Function
		orderedCrossOver 							::Function
		partialMatchCrossOver					::Function
		cycleCrossOver								::Function
		
		displacementMutation					::Function
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
			this.data = [population(M, Nbins) for i=1:N]
			this.N = N
			this.NSamples = 0

			##################################################################################################################### Functions
			#.................................................................................................................... Evaluation
			function evaluate(data::Array{Float32,1})
				N = length(data)
				No2 = Int64(N/2)
				psd = zeros(Float32,No2)
				psd = ((abs( fft(data) )./No2 )[1:No2]).^2;
				s = std(psd)
				return s
			end
			function evaluate(ind::Int64)
				N = length(this.data[ind].x)
				No2 = Int64(N/2)
				psd = zeros(Float32,No2)
				data = this.data[ind].x
				psd = ((abs( fft(data) )./No2 )[1:No2]).^2;
				s = std(psd)
				this.data[ind].score = s
				return s
			end
			this.evaluate = evaluate
			this.evaluateAll = function()
				for k=1:this.N
					this.data[k].score= this.evaluate(k)
				end
			end	
			this.repopulateAll = function(exceptID::Int64)
				for k=1:(this.N)
					if k!=exceptID
						this.data[k].repopulate()
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
			function getBest(indA::Int64, indB::Int64)
				dataA = this.data[indA].x
				dataB = this.data[indB].x
				ScoreA = this.evaluate(dataA)
				ScoreB = this.evaluate(dataB)
 				if ScoreA > ScoreB
					this.bestId = B
					this.S = ScoreB
					return indB, ScoreB,dataB
				else
					this.bestId = indA
					this.S = ScoreA
					return indA, ScoreA,dataA
				end	
			end
			function getBest()
				score = this.data[1].score
				ind = 1
				for k=2:this.N
					if this.data[k].score < b_score
						score = this.data[k].score
						ind = k
					end
				end
				return ind, score, this.data[ind].x
			end
			this.getBest = getBest



			#.................................................................................................................... Selection
			
			
			function rouletteWheelSelection(duplicates=true)
				this.evaluateAll()
				S = [this.data[k].score for k=1:this.N]
				S = cumsum(S./sum(S))
				p1_ids = [findfirst(S.>=rand()) for k=1:this.N]
				p2_ids = [findfirst(S.>=rand()) for k=1:this.N]
				if !(duplicates)
					for k=1:this.N
						while p1_ids[k] == p2_ids[k]
							p2_ids[k] = findfirst(S.>=rand())
						end
					end
				end
				return p1_ids, p2_ids
			end
			function rankSelection(duplicates=true)
				this.evaluateAll()
				S = [this.data[k].score for k=1:this.N]
				Sind = sortperm(S)
				p1_ids =[ Sind[rand([1:this.N;])] for k=1: this.N]
				p2_ids =[ Sind[rand([1:this.N;])] for k=1: this.N]
				if !(duplicates)
					for k=1:this.N
						while p1_ids[k] == p2_ids[k]
							p2_ids[k] = Sind[rand([1:this.N;])]
						end
					end
				end
				return p1_ids, p2_ids
			end
			function tournamentSelection(TournamentSize::Int64, duplicates=true)
				p1_ids = zeros(Int64,this.N)
				p2_ids = zeros(Int64,this.N)
				if TournamentSize<= this.N
					for k=1:this.N
						ind = [rand([1:this.N;]) for k=1:TournamentSize]
						S = [this.evaluate(ind[k]) for k=1:TournamentSize]
						p1_ids[k]= findmax(S)[2]
						ind = [rand([1:this.N;]) for k=1:TournamentSize]
						S = [this.evaluate(ind[k]) for k=1:TournamentSize]
						p2_ids[k]= findmax(S)[2]
					end
					if !(duplicates)
						for k=1:this.N
							while p1_ids[k] == p2_ids[k]
								ind = [rand([1:this.N;]) for k=1:TournamentSize]
								S = [this.evaluate(ind[k]) for k=1:TournamentSize]
								p2_ids[k]= findmax(S)[2]
							end
						end
					end
				else
					println("Tournament Size > number of populations")
				end
				return p1_ids, p2_ids
			end
			
			this.rouletteWheelSelection = rouletteWheelSelection
			this.rankSelection = rankSelection
			this.tournamentSelection = tournamentSelection
			
			
			
			#.................................................................................................................... CrossOver
			function order1XO(ind1::Array{Int64,1},ind2::Array{Int64,1}, p1::Array{Float32,1}, p2::Array{Float32,1})
				ini_Pt = rand([1:(length(ind1) - 1 );]);
				end_Pt = rand([(ini_Pt + 1):(length(ind1));]);
				childVec= zeros(Float32,length(p1))
				childVec_ind = zeros(Int64, length(p1))
				p1_ind = [find(p1[k].== p1[ind1])[1] for k=1:length(p1)];
				p2_ind = [find(p2[k].== p2[ind2])[1] for k=1:length(p2)];
				childVec_ind[ini_Pt:end_Pt] = p1_ind[ini_Pt:end_Pt]
				childVec[ini_Pt:end_Pt] = p1[ini_Pt:end_Pt]
				lind = end_Pt+1
				for k=(end_Pt+1):length(ind1)
					if any(p2_ind[k].== childVec_ind) == false
						if lind == (length(ind1)+1)
							lind = 1
						end
						childVec_ind[lind] = p2_ind[k]
						childVec[lind] = p2[ p2_ind[k] ]
						lind += 1
					end
				end	
				for k=1:end_Pt
					if any(p2_ind[k].== childVec_ind) == false
						if lind == (length(ind1)+1)
							lind = 1
						end
						childVec_ind[lind] = p2_ind[k]
						childVec[lind] = p2[ p2_ind[k] ]
						lind += 1
					end
				end	
				return childVec
			end
			function orderOneCrossOver(p1_ids::Array{Int64,1},p2_ids::Array{Int64,1})
				child1 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				child2 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				for k=1:this.N
					p1_ind = p1_ids[k]
					p2_ind = p2_ids[k]
					p1 = this.data[p1_ind].x
					p2 = this.data[p2_ind].x
					ind1 = sortperm(p1)
					ind2 = sortperm(p2)
					child1.data[k].x = order1XO(ind1,ind2, p1,p2)
					child2.data[k].x = order1XO(ind2,ind1, p2,p1)
				end
				return child1,child2
			end
			function PMX(ind1::Array{Int64,1},ind2::Array{Int64,1}, p1::Array{Float32,1}, p2::Array{Float32,1})
				ini_Pt = rand([1:(length(ind1) - 1 );]);
				end_Pt = rand([(ini_Pt + 1):(length(ind1));]);
				childVec = zeros(length(p1))
				childVec[ini_Pt:end_Pt] = p1[ini_Pt:end_Pt]
				p1_ind = [find(p1[k].== p1[ind1])[1] for k=1:length(p1)];
				p2_ind = [find(p2[k].== p2[ind2])[1] for k=1:length(p2)];
				mask1 = zeros(Int64,length(p1));
				mask1[ini_Pt:end_Pt] = 1;
				childVec_ind = zeros(Int64,length(p1));
				childVec_ind[ini_Pt:end_Pt]= p1_ind[ini_Pt:end_Pt];
										
				for k = ini_Pt:end_Pt
					if any( p2_ind[k].==childVec_ind)==false
						i = p2_ind[k]
						j = p1_ind[k]
						rj = find(p2_ind.==j)
						if mask1[rj][1]==0
							childVec[rj] = p2[i]
							childVec_ind[rj] = i
							mask1[rj] = 1
						else
							newk = find(p2_ind.==j)
							jj = p1_ind[newk]
							rjj = find(p2_ind.==jj)
							childVec[rjj] = p2[i]
							childVec_ind[rjj] = i
							mask1[rjj] = 1
						end
					end
				end
				while( any(mask1.==0) ==true )
					k = findfirst(mask1.==0)
					childVec[k] = p2[k]
					childVec_ind[k] = p2_ind[k]
					mask1[k] = 1
				end
				return childVec
			end
			function partialMatchCrossOver(p1_id::Int64,p2_id::Int64)
				return partialMatchCrossOver([p1_id],[p2_id])
			end
			function partialMatchCrossOver(p1_ids::Array{Int64,1},p2_ids::Array{Int64,1})
				child1 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				child2 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				for k=1:length(parent1_indexes)
					p1_ind = p1_ids[k]
					p2_ind = p2_ids[k]
					p1 = this.data[p1_ind].x
					p2 = this.data[p2_ind].x
					
					ind1 = sortperm(p1)
					ind2 = sortperm(p2)
					child1.data[k].x = PMX(ind1,ind2,p1,p2)
					child2.data[k].x = PMX(ind2,ind1,p2,p1)
				end
				return child1,child2
			end	
			function cycleXO(ind1::Array{Int64,1},ind2::Array{Int64,1}, p1::Array{Float32,1}, p2::Array{Float32,1})
				k_start = 1
				k_ind = 1
				seq1 = []
				seq1 = convert(Array{Int64,1},seq1)
				seq1_val = []
				p1_ind = [find(p1[k].== p1[ind1])[1] for k=1:length(p1)];
				p2_ind = [find(p2[k].== p2[ind2])[1] for k=1:length(p2)];
				childVec1 = zeros(Float32,length(p1))
				childVec2 = zeros(Float32,length(p1))
				pos = 1
				pos_ini = 1
				
				while length(seq1) <length(p1_ind)
					while true
						val_ind = find(p2_ind[pos] .== p1_ind)[1]
						seq1 = [seq1, val_ind]
						childVec1[val_ind] =  p1[p1_ind[val_ind]]   #p1[ p1_ind[val_ind] ]
						childVec2[val_ind] =  p2[p2_ind[val_ind]]
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
					while true
						val_ind = find(p1_ind[pos] .== p2_ind)[1]
						seq1 = [seq1, val_ind]
						childVec1[val_ind] = p2[p2_ind[val_ind]]
						childVec2[val_ind] = p1[p1_ind[val_ind]] 
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
				end
				return childVec1, childVec2
			end
			function cycleCrossOver(p1_ids::Array{Int64,1},p2_ids::Array{Int64,1})
				child1 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				child2 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				for k=1:this.N
					p1_ind = p1_ids[k]
					p2_ind = p2_ids[k]
					p1 = this.data[p1_ind].x
					p2 = this.data[p2_ind].x
					ind1 = sortperm(p1)
					ind2 = sortperm(p2)
					child1.data[k].x, child2.data[k].x = cycleXO(ind1,ind2, p1,p2)
				end
				return child1,child2
			end
			function positionBasedXO(ind1::Array{Int64,1},ind2::Array{Int64,1}, p1::Array{Float32,1}, p2::Array{Float32,1})
				p1_ind = [find(p1[k].== p1[ind1])[1] for k=1:length(p1)];
				p2_ind = [find(p2[k].== p2[ind2])[1] for k=1:length(p2)];
				#println("p1 = $(p1')")
				#println("p2 = $(p2')")
				#println("p1_ind = \t $(p1_ind')")
				#println("p2_ind = \t $(p2_ind')")
				childVec1 = zeros(Float32,length(p1))
				mask = rand([0:1;],length(p1))
				maskind= [p2_ind[k] for k=find(mask.==1)]
				oldmask= maskind
				childVec1 = mask.*p2
				p_i = 1
				c_i = 1
				
				#println("mask = \t\t $(mask')")
				#println("childVec1 = $(childVec1')")
				
				while c_i <= length(mask)
					#println("c_i=$(c_i) *********** mask[c_i] = $(mask[c_i])")
					if (mask[c_i]) == 0
						if (any(p1_ind[p_i].==oldmask)==false)
							oldmask = [oldmask, p1_ind[p_i]]
							childVec1[c_i] = p1[ p_i ]
							#println("^^^ c_i = $c_i \t p_i = $p_i \t childVec1 = $(childVec1') \t oldmask = $(oldmask')")
						else
							while (any(p1_ind[p_i].==oldmask)==true)
								p_i += 1
							end
							oldmask = [oldmask, p1_ind[p_i]]
							childVec1[c_i] = p1[ p_i ]
							#println("___ c_i = $c_i \t p_i = $p_i \t childVec1 = $(childVec1') \t oldmask = $(oldmask')")
						end
						p_i += 1
					end
					c_i += 1
					
				end
				
				return childVec1	
				
			end
			function positionBasedCrossOver(p1_ids::Array{Int64,1},p2_ids::Array{Int64,1})
				child1 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				child2 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				for k=1:this.N
					p1_ind = p1_ids[k]
					p2_ind = p2_ids[k]
					p1 = this.data[p1_ind].x
					p2 = this.data[p2_ind].x
					ind1 = sortperm(p1)
					ind2 = sortperm(p2)
					child1.data[k].x = positionBasedXO(ind1,ind2, p1,p2)
					child2.data[k].x = positionBasedXO(ind2,ind1, p2,p1)
				end
				return child1,child2
			end
			
			
			
			this.cycleCrossOver = cycleCrossOver
			this.orderOneCrossOver= orderOneCrossOver
			this.partialMatchCrossOver = partialMatchCrossOver
			this.positionBasedCrossOver = positionBasedCrossOver
			
			
			
			
			#.................................................................................................................... Mutation
			function insertMutation(prob_mutation::Float64)
				child1 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				for k=1:this.N
					p1 = this.data[k].x
					if rand()<= prob_mutation
						pos1 = rand([1:(length(p1) - 2);])
						pos2 = rand([pos1+2:length(p1);])
						#println("p1 = $(p1)	\t pos1 = $(pos1) \t pos2 = $(pos2) ")
						insert!(p1, pos1, splice!(p1,pos2))
						#println("p1 = $(p1)	\t pos1 = $(pos1) \t pos2 = $(pos2) ")
					end
					child1.data[k].x = p1
				end
				return child1
			end
			function swapMutation(prob_mutation::Float64)
				child1 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				for k=1:this.N
					p1 = this.data[k].x
					if rand()<= prob_mutation
						pos1 = rand([1:(length(p1) - 1);])
						pos2 = rand([(pos1+1):length(p1);])
						temp = p1[pos1]
						p1[pos1] = p1[pos2]
						p1[pos2]=temp
					end
					child1.data[k].x = p1
				end
				return child1
			end
			function inversionMutation(prob_mutation::Float64)
				child1 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				for k=1:this.N
					p1 = this.data[k].x
					if rand()<= prob_mutation
						pos1 = rand([1:(length(p1) - 1);])
						pos2 = rand([(pos1+1):length(p1);])
						p1[pos1:1:pos2] = p1[pos2:-1:pos1]
					end
					child1.data[k].x = p1
				end
				return child1
			end
			function scrambleMutation(prob_mutation::Float64)
				child1 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				for k=1:this.N
					p1 = this.data[k].x
					if rand()<= prob_mutation
						pos1 = rand([1:(length(p1) - 1);])
						pos2 = rand([(pos1+1):length(p1);])
						p1[pos1:1:pos2] = shuffle(p1[pos1:1:pos2])
					end
					child1.data[k].x = p1
				end
				return child1
			end
			function displacementMutation(l::Int64, prob_mutation::Float64)
				child1 = GAmodel(this.N,this.data[1].N,this.data[1].Nbins)
				temp = zeros(Float32, l)
				for k=1:this.N
					p1 = this.data[k].x
					if rand()<= prob_mutation
						pos1 = rand([1:(length(p1) - 1 - l);])
						temp =splice!(p1,pos1:(pos1+l))
						pos2 = rand([1:(length(p1)-1;])
						p1 = [p1[1:pos2], temp, p1[(pos2+1):end]]
					end
					child1.data[k].x = p1
				end
				return child1
			end
			
			
			this.displacementMutation = displacementMutation
			this.scrambleMutation = scrambleMutation
			this.inversionMutation =inversionMutation
			this.swapMutation = swapMutation
			this.insertMutation = insertMutation
			this.exchangeMutation = function(p1_id::Int64,p2_id::Int64, prob_mutation::Float64)
				p1 = this.data[ p1_id ].x 
				p2 = this.data[ p2_id ].x
				if rand() < prob_mutation					
					A = rand([1:(length(p1)-1);])
					B = rand([(A+1):length(p1);])
					temp = p1[A]
					p1[A] = p1[B]
					p1[B] =temp
					this.data[ p1_id ].x = p1
				end
				if rand() < prob_mutation	
					A = rand([1:(length(p1)-1);])
					B = rand([(A+1):length(p1);])
					temp = p2[A]
					p2[A] = p2[B]
					p2[B] =temp
					this.data[ p2_id ].x = p2
				end
				return p1, p2
			end
			
			
			this.elitistSelection = function()
				this.evaluateAll()
				k_prime, _, _ = this.getBest()
				#list = symdiff(k_prime,[1:this.N;])
				list = [1:this.N;]
				#parent1_indexes = [k_prime for k=1:(this.N-1)]
				p1_ids = [k_prime for k=1:(this.N)]
				p2_ids = list
				return convert(Array{Int64,1},p1_ids),convert(Array{Int64,1},p2_ids)
			end
			this.elitistOrderedCrossOver = function(p1_ids::Array{Int64,1},p2_ids::Array{Int64,1})
				for k=1:length(parent1_indexes)
					p1_ind = p1_ids[k]
					p2_ind = p2_ids[k]
					
					p1 = copy(this.data[ p1_ind ].x) 
					p2 = copy(this.data[ p2_ind ].x)

					if p1_ind != p2_ind
						ind1 = sortperm(p1)
						ind2 = sortperm(p2)

						cut_A = rand([1:(length(p1)-1);])
						cut_B = rand([(cut_A+1):length(p1);])

						childVec1 = zeros(Float32,size(p1))
						childVec2 = zeros(Float32,size(p1))
				
						mask1 = ones(Int64,size(p1))
						mask2 = ones(Int64,size(p1))
				
						childVec1[cut_A:cut_B] = p1[cut_A:cut_B]
						childVec2[cut_A:cut_B] = p2[cut_A:cut_B]
				
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
							childVec1[k0] = temp1[k_ind]
							childVec2[k0] = temp2[k_ind]
							k_ind +=1
						end
						for k0 = (cut_B+1):length(p1)
							childVec1[k0] = temp1[k_ind]
							childVec2[k0] = temp2[k_ind]
							k_ind +=1
						end

					
						##println("eval(p1) $(this.evaluate(p1))  vs  eval(childVec1) $(this.evaluate(childVec1)) \t \t eval(p2) $(this.evaluate(p2)) \t vs  eval(childVec2) $(this.evaluate(childVec2))")
						#this.data[ pop_ind1 ].x = childVec1
						#this.data[ pop_ind2 ].x = childVec2
					end
				end
				#return pop_ind1,pop_ind2, this.data[ pop_ind1 ].x, this.data[ pop_ind2 ].x
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





