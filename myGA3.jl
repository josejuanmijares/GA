
module myGA3
	using Compat

	export SuperJuice

	@compat typealias TestableNumbers Union{Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Int128, UInt128, Float16, Float32, Float64}
	
	type SuperJuice{T<:TestableNumbers}
		x													::SharedArray{T,2}							# population elements
		N													::Int64													# length of the population
		M													::Int64													# Number of populations
		scores										::SharedArray{T,1}							# fitness
		NSamples									::Int64													# total number of samples
	
		function SuperJuice(N::Int64 = 1024, M::Int64 = 8, iter = 100, mypids = workers())
			this = new()
			this.N = N
			this.M = M
			this.NSamples = 0
			this.x = SharedArray(T,(N,M), init=S->S[localindexes(S)]=rand(T,length([localindexes(S);])), pids = mypids)
			this.scores = SharedArray(T, M, init= S2->S2[localindexes(S2)]=zeros(T,length([localindexes(S2);])), pids = mypids)
			if T<:AbstractFloat
				## EHE correction part
				np = nprocs()
				i = 1
				nextidx() = (idx=i; i+=1; idx)
				@sync begin
					for p in mypids
						if p != myid() || np == 1
							@async begin
								while true
									idx = nextidx()
									if idx > this.M
										break
									end
									remotecall_wait(p, EHEfast_0to1, this.x, N, idx)
									remotecall_wait(p, doEvaluateAll, this.x, this.scores, this.N, idx)
								end
							end
						end
					end
				end

				child = SharedArray(T,(N,M))
				i=1 ; nextidx() = (idx=i; i+=1; idx)
				p=-1
				@sync begin
					for p in mypids
						if p != myid() || np == 1
							@async begin
								while true
									idx = nextidx()
									if idx > this.M
										break
									end
									remotecall_wait(p, doSuperJuice, this.x, this.scores, child, N, M, idx, iter)
								end
							end
						end
					end
				end
			end
			return this
		end	
	end
	
	function EHEfast_0to1{T}(xin::SharedArray{T,2},N::Int64, idx::Int64)
		ind = sortperm(xin[:,idx])
		ind2 = zeros(N)
		[ind2[ind[k]] = k for k=1:N]
		binsize = 2
		ratio = convert(T, binsize)/convert(T, N)
		offset = floor((ind2-1)/binsize)*ratio
		xin[:,idx] = mod(xin[:,idx],ratio) + offset
	end
	function evaluate{T}(S::Array{T,1})
		N = length(S)
		No2 = Int64(N/2)
		psd = zeros(T,No2+1)
		psd = (1/N)*( abs( fft(S)[1:No2] ) .^2)
		return std(psd)
	end
	function doEvaluateAll{T}(S::SharedArray{T,2}, S2::SharedArray{T,1}, N::Int64, idx::Int64)
		No2 = Int64(N/2)
		psd = zeros(T,No2+1)
		psd = (1/N)*( abs( fft(S[:,idx])[1:No2] ) .^2)
		S2[idx] =  std(psd)
	end
	
	function doRouletteWheelSelection{T}(S::Array{T,1}, duplicates::Bool = false)
		p1_id = findfirst(S.>=rand(T,1))
		p2_id = findfirst(S.>=rand(T,1))
		if !(duplicates)
			while p1_id == p2_id
				p2_id = findfirst(S.>=rand(T,1))
			end
		end
		return p1_id, p2_id
	end

	function doOrder1X0_2{T}(p1_id::Int64,p2_id::Int64, S::SharedArray{T,2}, N::Int64, childVec::SharedArray{T,2}, idx::Int64)
		p1_ind = shuffle([1:N;])
		p2_ind = shuffle([1:N;])
		childVec_ind = zeros(Int64,N)
		ini_Pt = rand( [1:(N - 1 );]);
		end_Pt = rand( [(ini_Pt + 1):N;]);
		A1 = zeros(Int64, end_Pt-ini_Pt + 1 )
		A1 = p1_ind[ini_Pt:end_Pt]
		A0 = setdiff(p2_ind, A1 )[1:(ini_Pt-1)]
		A2 = setdiff(p2_ind, [A1;A0] )
		childVec_ind = [A0; A1; A2]
		childVec[:,idx] = S[ childVec_ind, p1_id ]
	end

	function doShuffleMutation{T}(child1::SharedArray{T,2}, idx::Int64, N::Int64, prob_mutation::Float64)
		if rand(Float64,1)<= prob_mutation
			child1[:,idx] = shuffle(child1[:,idx])
		end
	end

	function muReplacement{T}(mu::Int64, mu2::Int64 , qTournaments::Int64, child1::SharedArray{T,2}, N::Int64, M::Int64)
		twoN = 2*M
		c1_scores = SharedArray(Float32, M, init= S2->S2[localindexes(S2)]=zeros(Float32,length([localindexes(S2);])), pids=workers());
		p = population(N,M);
		this.evaluateAll(child1,c1_scores)

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








	function doSuperJuice{T}(x::SharedArray{T,2}, scores::SharedArray{T,1}, child::SharedArray{T,2}, N::Int64, M::Int64,idx::Int64, k_stop::Int64)
		
		p1_id :: Int64
		p2_id :: Int64
		#doEvaluateAll(x,scores,N,idx)
		p1_id, p2_id = doRouletteWheelSelection(cumsum(scores./sum(scores)))
		doOrder1X0_2(p1_id,p2_id,x,N,child,idx)
		doShuffleMutation(child,idx,N,0.1)
		
		
	end

end



# 		println("scores[$(idx)] = $(scores[idx])")
#
# #			k = 1
# # 		#csvfile = open("data.csv","w")
# # 		while k <= k_stop
# # 			parentA, parentB = g.rouletteWheelSelection(false);
# # 			#println("$k \t\t\t scores -> $(g.getScores())" )
# # 			childA, childB = g.orderOneCrossOver_2(parentA, parentB);
# # 			g.shuffleMutation(1.0, childA, childB)
# # 			g.muReplacement(Int64(g.M/4), Int64(g.M/2), 10, childA, childB)
# # 			#g.replaceWorst(Int64(g.N/2),childA,childB)
# # 			#println("$k \t\t\t scores -> $(g.getScores())" )
# # 			#println("$k \t\t\t best -> $(g.getBest())" )
# # 			#println("-----------------------------------------------------------------------------------------------------------")
# # 			#savevec = [k; g.getBest()[1]; g.getBest()[2]; g.getScores()]
# # 			#write(csvfile, join(savevec,","), "\n")
# # 			k+=1;
# # 		end
# # 		#println(" k= $k \n g[1] = $(g.data[g.getBest()[2]].x')")
# # 		#close(csvfile)
# # 		return g.x[:, g.getBest()[2]]
# 	end
#
#
################################################
		#
		# 		function outputNumbers(N::Int64)
		# 			if N > length(this.buffer)
		# 				this.appendNumbers(N)
		# 			end
		# 			out = [pop!(this.buffer) for k=1:N]
		# 			if length(this.buffer)==0
		# 				this.reloadNumbers()
		# 			end
		# 			return out
		# 		end
		#
		# 		function appendNumbers(N::Int64)
		#
		# 			while length(this.buffer) < N
		# 				this.reloadNumbers()
		# 			end
		#
		# 		end
		#
		# 		function reloadNumbers()
		#
		# 			this.buffer = [this.buffer; copy(this.doSuperJuice(this.g,100))]
		# 		end
		#
		# 		function doSuperJuice(g::population, k_stop::Int64)
		# 			g.evaluateAll()
		# 			k = 1
		# 			#csvfile = open("data.csv","w")
		# 			while k <= k_stop
		# 				parentA, parentB = g.rouletteWheelSelection(false);
		# 				#println("$k \t\t\t scores -> $(g.getScores())" )
		# 				childA, childB = g.orderOneCrossOver_2(parentA, parentB);
		# 				g.shuffleMutation(1.0, childA, childB)
		# 				g.muReplacement(Int64(g.M/4), Int64(g.M/2), 10, childA, childB)
		# 				#g.replaceWorst(Int64(g.N/2),childA,childB)
		# 				#println("$k \t\t\t scores -> $(g.getScores())" )
		# 				#println("$k \t\t\t best -> $(g.getBest())" )
		# 				#println("-----------------------------------------------------------------------------------------------------------")
		# 				#savevec = [k; g.getBest()[1]; g.getBest()[2]; g.getScores()]
		# 				#write(csvfile, join(savevec,","), "\n")
		# 				k+=1;
		# 			end
		# 			#println(" k= $k \n g[1] = $(g.data[g.getBest()[2]].x')")
		# 			#close(csvfile)
		# 			return g.x[:, g.getBest()[2]]
		# 		end
		#
		# 		this.outputNumbers = outputNumbers
		# 		this.appendNumbers = appendNumbers
		# 		this.reloadNumbers = reloadNumbers
		# 		this.doSuperJuice = doSuperJuice
		#
		# 		this.buffer = copy(this.doSuperJuice(this.g,100));

		#		return this
		#	end

