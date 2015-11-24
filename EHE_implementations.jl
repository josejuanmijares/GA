function EHE(xin, Nbins)
	l = length(xin)
	x0=xin;
	ind0 =[1:l;]
	ind1 = sortperm(x0)
	x1 = floor(ind1 / round(l/Nbins))
	return x1
end

function EHEfast0(xin,Nbins,typein)
	ind = sortperm(xin)
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

function EHEfast1(xin,Nbins,typein)
	ind = sortperm(xin)
	l = length(xin)
	binsize = round(l/Nbins);		
	offset = length(bits(xin[1])) - log2(Nbins) 
	MSB = zeros(typein,l)
	LSB = repmat([~(typein(Nbins-1) << Int(offset))],l)
	xout = zeros(typein, l)
	MSB = convert(Array{typein,1}, floor(ind./float(binsize))).<< Int(offset)
	xout = (xin & MSB) | (xin & LSB)
	return xout
end