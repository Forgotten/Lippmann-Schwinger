# Helper functions 


# reference Helmholtz solution for a centered Gaussian 
@inline function solRefHelmholtz(x::Array{Float64,1},y::Array{Float64,1},z::Array{Float64,1}, sigma::Float64, k::Float64)
  r  = sqrt( x.^2 + y.^2 + z.^2)
  u  = (exp(-sigma^2*k^2/2)./(4*pi.*r)).*(real( exp(-1im.*k.*r).*erf( (2*sigma^2*1im*k - 2.*r)./(2*sqrt(2*sigma^2) ) )) - 1im*sin(k.*r))
  return u
end


@inline function Gtruncated2D{T<:Number,N}(L::Float64,k::Float64,s::Array{T,N})
    return (1 + (1im*pi/2*L*hankelh1(0,L*k))*(s.*besselj(1,L*s)) - (1im*pi/2*L*k*hankelh1(1,L*k))*besselj(0,L*s)  )./(s.^2 - k^2)
end


@inline function Gtruncated3D{T<:Number,N}(L::Float64,k::Float64,s::Array{T,N})
    return (-1 + exp(1im*L*k).*( cos(L*s) - (1im*k*L*sinc(L*s/pi)) ) )./( (k^2-s.^2))
end

@inline function Gtruncated3D(L::Float64,k::Float64,s::Float64)
    return (-1 + exp(1im*L*k)*( cos(L*s) - (1im*k*L*sinc(L*s/pi)) ))/( (k^2-s^2))
end