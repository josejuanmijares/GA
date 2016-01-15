# push!(LOAD_PATH, pwd()); LOAD_PATH=unique(LOAD_PATH);using DummyTest
# workspace(); reload("DummyTest"); using  DummyTest;

@everywhere using ExactHistEq

module myGA2

	export population,  GAmodel, SuperJuice, startSuperJuice, rand, mypmap

	import ExactHistEq.EHEfast3
	import Goertzel_functions.goertzel, Goertzel_functions.online_variance
	import Base.rand
	
	defaultSize = Int64(2^10)
	
	type population
		x													::SharedArray{Float32,2}				# population elements
		N													::Int64													# length of the population
		M													::Int64													# Number of populations
		scores										::SharedArray{Float32,1}				# fitness
		NSamples									::Int64													# total number of samples
		
		repopulate								::Function
		evaluateAll								::Function
		getBest										::Function
		getScores									::Function
		rouletteWheelSelection		::Function
		orderOneCrossOver_2				::Function
		insertMutation						::Function
		muReplacement							::Function
		shuffleGA									::Function
		
		function population(N::Int64, M::Int64)
			this = new()
			this.N = N
			this.M = M
			this.NSamples = 0
			this.x = SharedArray(Float32,(N,M), init=S->S[localindexes(S)]=rand(length([localindexes(S);])), pids=workers());
			this.scores = SharedArray(Float32, M, init= S2->S2[localindexes(S2)]=zeros(Float32,length([localindexes(S2);])), pids=workers());
			function doRandomNumbers(S::SharedArray{Float32,2}, N::Int64)
				ind = localindexes(S)
				Sd = sdata(S)
				
				for i=(ind.start):N:((ind.stop))
					Sd[i:(i+N-1)] = EHEfast3(Sd[i:(i+N-1)])
				end
				
			end
			function initRandomNumbers(S::SharedArray{Float32,2}, N::Int64)
				np = nprocs() 
				@sync begin
					for p=1:np
						if p != myid() || np == 1
							@async begin
								remotecall_fetch(p,doRandomNumbers,S, N)
							end
						end
					end
				end	
			end			
			initRandomNumbers(this.x,this.N)
			this.repopulate = function( )
				this.x = SharedArray(Float32,(N,M), init=S->S[localindexes(S)]=rand(length([localindexes(S);])), pids=workers());
				initRandomNumbers(this.x,this.N)
			end
			
			function evaluate(S::Array{Float32,1})
				N = length(S)
				No2 = Int64(N/2)
				psd = zeros(Float32,No2+1)
				psd = (1/N)*( abs( fft(S)[1:No2] ) .^2)
				return std(psd)
			end
				
			
			function doEvaluateAll(S::SharedArray{Float32,2}, S2::SharedArray{Float32,1},  N::Int64)
				ind = localindexes(S)
				Sd = sdata(S)
				ind2 = localindexes(S2)
				Ss = sdata(S2)
				No2 = Int64(N/2)
				j= ind2.start
				for i=(ind.start):N:((ind.stop))
					psd = zeros(Float32,No2+1)
					psd = (1/N)*( abs( fft(Sd[i:(i+N-1)] )[1:No2] ) .^2)
					#psd = ((abs( fft(Sd[i:(i+N-1)]) )./No2 )[1:No2]).^2;
					#println("mean=$(mean(psd) ) \t  std=$(std(psd))")
					#Ss[j] =  sqrt((mean(psd) -1 )^2  + (std(psd))^2)
					Ss[j] =  std(psd)
					j+=1
				end
			end	
			function evaluateAll()
				np = nprocs() 
				@sync begin
					for p=1:nthis.M
						if p != myid() || np == 1
							@async begin
								remotecall_fetch(p,doEvaluateAll, this.x, this.scores, this.N )
							end
						end
					end
				end	
			end	
			function evaluateAll(S::SharedArray{Float32,2}, scores::SharedArray{Float32,1})
				np = nprocs() 
				@sync begin
					for p=1:this.M
						if p != myid() || np == 1
							@async begin
								remotecall_fetch(p,doEvaluateAll, S, scores, this.N )
							end
						end
					end
				end
			end
			this.evaluateAll = evaluateAll
			
			function getBest()
				this.evaluateAll()
				return findmin(this.scores)
			end
			function getScores()
				this.evaluateAll()
				return sdata(this.scores)
			end
			
			this.getBest = getBest
			this.getScores =getScores
			
			function rouletteWheelSelection(duplicates=true)
				this.evaluateAll()
				S = this.getScores()
				S = cumsum(S./sum(S))
				p1_ids = [findfirst(S.>=rand()) for k=1:this.M]
				p2_ids = [findfirst(S.>=rand()) for k=1:this.M]
				if !(duplicates)
					for k=1:this.M
						while p1_ids[k] == p2_ids[k]
							p2_ids[k] = findfirst(S.>=rand())
						end
					end
				end
				return convert(Array{Int64,1},p1_ids), convert(Array{Int64,1},p2_ids)
			end
			this.rouletteWheelSelection = rouletteWheelSelection
			function doOrder1X0_2(p1_id::Int64,p2_id::Int64, S::SharedArray{Float32,2}, N::Int64)
				Sd = sdata(S)
				p1_ind = zeros(Int64,N)
				p2_ind = zeros(Int64,N)
				childVec_ind = zeros(Int64,N)
				childVec = zeros(Float32,N)
				setindex!(p1_ind,[1:N;], sortperm( Sd[:,p1_id] ))
				setindex!(p2_ind,[1:N;], sortperm( Sd[:,p2_id] ))
				ini_Pt = rand([1:(N - 1 );]);
				end_Pt = rand([(ini_Pt + 1):N;]);
				
				childVec_ind[ini_Pt:end_Pt] = p1_ind[ini_Pt:end_Pt]
				lind = end_Pt+1
				for k=lind:N
					if any(p2_ind[k].== childVec_ind) == false
						if lind == (N+1)
							lind = 1
						end
						childVec_ind[lind] = p2_ind[k]
						lind += 1
					end
				end
				
				for k=1:end_Pt
					if any(p2_ind[k].== childVec_ind) == false
						if lind == (N+1)
							lind = 1
						end
						childVec_ind[lind] = p2_ind[k]
						lind += 1
					end
				end
				childVec = Sd[childVec_ind,p1_id]
				return childVec
			end
			function orderOneCrossOver_2(p1_ids::Array{Int64,1},p2_ids::Array{Int64,1})
				#child1 = population(this.N,this.M)
				#child2 = population(this.N,this.M)
				
				child1=SharedArray(Float32,(N,M), init=false, pids=workers());
				child2=SharedArray(Float32,(N,M), init=false, pids=workers());
				
				n = length(p1_ids)
				np = nprocs()
				i=1
				nextidx() = (idx=i; i+=1; idx)
				@sync begin
					for p=1:np
						if p != myid() || np == 1
							@async begin
								while true
								idx = nextidx()
									if idx > n
										break
									end
									child1[:,idx] = remotecall_fetch(p, doOrder1X0_2, p1_ids[idx],p2_ids[idx], this.x, this.N)
									child2[:,idx] = remotecall_fetch(p, doOrder1X0_2, p2_ids[idx],p1_ids[idx], this.x, this.N)
								end
								#remotecall_fetch(p,doEvaluateAll, this.x, this.scores, this.N,  )
							end
						end
					end
				end
				return child1,child2	
			end				
			this.orderOneCrossOver_2 =orderOneCrossOver_2
			
			function doInsertMutation(child1::SharedArray{Float32,2}, idx::Int64, N::Int64,prob_mutation::Float64)
				Sd = sdata(child1)
				temp = copy(Sd[:,idx])
				if rand()<= prob_mutation
					
					temp = shuffle(temp)
					
					# pos1 = rand([1:(N - 2);])
	# 				pos2 = rand([pos1+2:N;])
	# 				insert!(temp, pos1, splice!(temp,pos2))
					#println("p1 = $(p1)	\t pos1 = $(pos1) \t pos2 = $(pos2) ")
					#println("p1 = $(p1)	\t pos1 = $(pos1) \t pos2 = $(pos2) ")
				end
				Sd[:,idx]=temp
			end
			function insertMutation(prob_mutation::Float64, child1::SharedArray{Float32,2}, child2::SharedArray{Float32,2})
				np = nprocs()
				i=1
				nextidx() = (idx=i; i+=1; idx)
				@sync begin
					for p=1:np
						if p != myid() || np == 1
							@async begin
								while true
								idx = nextidx()
									if idx > this.M
										break
									end
									remotecall_fetch(p, doInsertMutation, child1, idx, this.N, prob_mutation)
									remotecall_fetch(p, doInsertMutation, child2, idx, this.N, prob_mutation)
								end
								#remotecall_fetch(p,doEvaluateAll, this.x, this.scores, this.N,  )
							end
						end
					end
				end
			end
			this.insertMutation = insertMutation
			
			function muReplacement(mu::Int64, mu2::Int64 , qTournaments::Int64, child1::SharedArray{Float32,2}, child2::SharedArray{Float32,2})
				M = this.M
				N = this.N
				twoN = 2*M
				
				#this.evaluateAll()
				c1_scores = SharedArray(Float32, M, init= S2->S2[localindexes(S2)]=zeros(Float32,length([localindexes(S2);])), pids=workers());
				c2_scores = SharedArray(Float32, M, init= S2->S2[localindexes(S2)]=zeros(Float32,length([localindexes(S2);])), pids=workers());
				p = population(N,M);
				this.evaluateAll(child1,c1_scores)
				this.evaluateAll(child2,c2_scores)
				
				scores0 = [ sdata(c1_scores); sdata(c2_scores) ]
				
				indexes = [1:twoN;]
				c = shuffle(indexes)

				wins = zeros(Int64,twoN)
				scores = zeros(twoN)
				scores = scores0[c]

				ct1 = c
				ct2 = circshift(ct1,1)
				for t=1:qTournaments
					for qind =1:length(c)
						ind1 = ct1[qind]
						ind2 = ct2[qind]
						if scores[ind1] >= scores[ind2]
							wins[ind1] += 1
						else
							wins[ind2] += 1
						end
					end
					ct2 = circshift(ct2,1)
				end

				wins_ind = sortperm(wins,rev=true)
				scores = this.getScores()
				scores_ind = sortperm(scores,rev=true)

				for k=1:mu
					i = c[wins_ind[k]]
					ki = scores_ind[k]
					if i <= M
						this.x[:,ki] = child1[:,i]
					else
						i = i - M
						this.x[:,ki] = child2[:,i]
					end
				end
				for k=(mu+1):mu2
					ki = scores_ind[k]
					this.x[:,ki] = p.x[:,k]
				end
			end
			this.muReplacement=muReplacement
			
			

			
			return this
		end
		function population(x::Array{Float32,1})
			this = new()
			this.x = x
			this.N = length(x)
			this.scores = 0.0f0
			this.NSamples = 0
			return this
		end
		function population()
			return population(defaultSize,8)
		end
	end
	

#
	type SuperJuice
		buffer											::Array{Float32,1}							# buffer values
		N														::Int64													# N values requested
		g														::population

		outputNumbers								::Function
		appendNumbers								::Function
		reloadNumbers								::Function
		doSuperJuice								::Function

		function SuperJuice()
			this =new()
			this.g = population(defaultSize,8)
			this.N = 0

			function outputNumbers(N::Int64)
				if N > length(this.buffer)
					this.appendNumbers(N)
				end
				out = [pop!(this.buffer) for k=1:N]
				if length(this.buffer)==0
					this.reloadNumbers()
				end
				return out
			end

			function appendNumbers(N::Int64)
				
				while length(this.buffer) < N
					this.reloadNumbers()
				end

			end

			function reloadNumbers()

				this.buffer = [this.buffer; copy(this.doSuperJuice(this.g,10000))]
			end

			function doSuperJuice(g::population, k_stop::Int64)
				g.evaluateAll()
				k = 1
				#csvfile = open("data.csv","w")
				while k <= k_stop
					parentA, parentB = g.rouletteWheelSelection(false);
					#println("$k \t\t\t scores -> $(g.getScores())" )
					childA, childB = g.orderOneCrossOver_2(parentA, parentB);
					g.insertMutation(1.0, childA, childB)
					g.muReplacement(Int64(g.M/4), Int64(g.M/2), 10, childA, childB)
					#g.replaceWorst(Int64(g.N/2),childA,childB)
					#println("$k \t\t\t scores -> $(g.getScores())" )
					println("$k \t\t\t best -> $(g.getBest())" )
					#println("-----------------------------------------------------------------------------------------------------------")
					#savevec = [k; g.getBest()[1]; g.getBest()[2]; g.getScores()]
					#write(csvfile, join(savevec,","), "\n")
					k+=1;
				end
				#println(" k= $k \n g[1] = $(g.data[g.getBest()[2]].x')")
				#close(csvfile)
				return g.x[:, g.getBest()[2]]
			end

			this.outputNumbers = outputNumbers
			this.appendNumbers = appendNumbers
			this.reloadNumbers = reloadNumbers
			this.doSuperJuice = doSuperJuice

			@time this.buffer = copy(this.doSuperJuice(this.g,10000));

			return this
		end
	end

	function startSuperJuice(SJ::SuperJuice,N::Int64)
		if length(SJ.buffer)<N
			SJ.appendNumbers(N)
		end
		return SJ.outputNumbers(N)
	end

	# isdefined(:SuperJuice)
	if method_exists(rand,(myGA2.SuperJuice,Int64)) == false
		rand(r::SuperJuice,N::Int64) = startSuperJuice(r,N)
	end
	
end





#=
function startSuperJuice(SJ::SuperJuice,N::Int64)
	if length(SJ.buffer)<N
		SJ.appendNumbers(N)
	end
	return SJ.outputNumbers(N)
end





	function SuperJuice(N::Int64)
		if N<=512
			buffer = zeros(Float32,N)
			g = GAmodel(16,N,Int64(N/2));
			g = doSuperJuice(g, 500)
			#println(g)
			#println(g.getBest())
			#error("hola")
			buffer = g.data[g.getBest()[2]].x
			return buffer
		end
	end
	function doSuperJuice(g::GAmodel, k_stop::Int64)
		g.evaluateAll();	
		k = 1
		#csvfile = open("data.csv","w")
		while k <= k_stop
			parentA, parentB = g.rouletteWheelSelection(false);
			childA, childB = g.orderOneCrossOver_2(parentA, parentB);
			childA = childA.insertMutation(0.1); childB = childB.insertMutation(0.1);
			g.muReplacement(Int64(4), Int64(g.N-2) , 10, childA, childB)
			#g.replaceWorst(Int64(g.N/2),childA,childB)
			println("$k \t\t\t best -> $(g.getBest())" )
			#savevec = [k; g.getBest()[1]; g.getBest()[2]; g.getScores()]
			#write(csvfile, join(savevec,","), "\n")
			k+=1;
		end
		#println(" k= $k \n g[1] = $(g.data[g.getBest()[2]].x')")
		#close(csvfile)
		return g;
	end
=#




