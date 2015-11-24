module ExactHistEq

export EHE, EHEfast0, EHEfast1

	function EHE(xin, Nbins)
		l = length(xin)
		x0=xin;
		ind0 =[1:l;]
		ind1 = sortperm(x0,rev=true)
		x1 = floor(ind1 / round(l/Nbins))
		return x1
	end
	
	function EHEfast0(xin,Nbins,typein)
		
		if Nbins < 2^8
			typein = UInt8
		elseif Nbins  < 2^16
			typein = UInt16
		elseif Nbins < 2^32
			typein = UInt32
		else
			typein = UInt64
		end
		
		ind = sortperm(xin,rev=true)
		l = length(xin)
		binsize = round(l/Nbins);		
		offset = length(bits(xin[1])) - log2(Nbins) 
		MSB = typein(0)
		LSB =  ~(typein(Nbins-1) << Int(offset))
		xout = zeros(typein, length(ind))
		for k=1:l
			MSB = typein(floor(float(ind[k])/float(binsize))) << Int(offset)
			xout[k] = (xin[k] & MSB) | (xin[k] & LSB)
		end
		return xout
	end
	
	function EHEfast1(xin,Nbins)
		
		if Nbins < 2^8
			typein = UInt8
		elseif Nbins  < 2^16
			typein = UInt16
		elseif Nbins < 2^32
			typein = UInt32
		else
			typein = UInt64
		end
		
		ind = sortperm(xin,rev=true)
		l = length(xin)
		binsize = round(l/Nbins);
		
		if issubtype(UInt,typeof(xin[1]))
			offset = length(bits(xin[1])) - log2(Nbins) 
			MSB = zeros(typein,l)
			LSB = repmat([~(typein(Nbins-1) << Int(offset))],l)
			xout = zeros(typein, l)
			MSB = convert(Array{typein,1}, floor(ind./float(binsize))).<< Int(offset)
			xout = (xin & MSB) | (xin & LSB)
		end
		
		if typeof(xin[1])==Float32
			binstep = Float32(1.0)/Float32(Nbins)
			xout = floor(convert(Array{Float32,1},ind-1)/Float32(binsize))*binstep + mod(xin,binstep) 
		end
		
		return xout
	end
end