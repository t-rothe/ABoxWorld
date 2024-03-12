
#module ABoxWorld
#    export nsboxes, wirings, conditions, games, sdp_conditions

if !isdefined(Main, :projectname)
    @error """Apparantly DrWatson is not loaded. Please include "using DrWatson" before using this code"""
elseif projectname() != "ABoxWorld"
    @error """This code is meant to be used in the ABoxWorld project, but found "$(projectname())". Please activate the ABoxWorld project by "@quickactivate ABoxWorld" before using this code"""
else
    @info "ABoxWorld project environment is loaded and active"
end

include("ABoxWorld_core.jl");

#Extensions, only for convenience:
include(srcdir("common_nsboxes.jl"));
include(srcdir("common_conditions.jl"));
include(srcdir("common_wirings.jl"));

#end