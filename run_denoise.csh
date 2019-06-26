#!/bin/tcsh -f

set args = args.denoise

echo -n '' > $args
foreach mask ('-repl' '-mask')
foreach smooth (0.1 0.2 0.5 1 2 5 10 20 50 99)
foreach thresh (2.0 2.5 3.0 3.5 4.0 5.0 7.0 9.9)
foreach scale (0.1 0.5 1 2 3 4 5 9 0.0)
foreach blur (01 05 11 21 31 51 71 99)
foreach amp (01 05 07 10 15 20 50)
    echo $mask -smooth $smooth -thresh $thresh -scale $scale -blur $blur -amp $amp >> $args
end
end
end
end
end
end

gshuf $args | xargs -n 11 ./denoise.py -in x.avi -train 65 -1x -dump 84,137,145,160
#gshuf $args | head | tr '\n' '\0' | xargs -n 1 -0 echo ./denoise.py -in x.avi -train 65 -1x -dump 84,137,145,160 ARGS
