#wald@moggy:/space/lander-vopat-animated$ ~/Projects/umesh/bin/umeshImportLanderFun3D /mnt/raid/wald/models/lander-small/geometry/dAgpu0145_Fa_me --scalars /mnt/raid/wald/models/lander-small/10000unsteadyiters/dAgpu0145_Fa_volume_data. -o /space/lander-vopat-animated/lander-small-merged.umesh
#File Info: 
#variables: rho u v w p turb1 vort_mag
#timeSteps: 6200 6400 6600 6800 7000 7200 7400 7600 7800 8000 8200 8400 8600 8800 9000 9200 9400 9600 9800 10000
~/Projects/umesh/bin/umeshImportLanderFun3D /mnt/raid/wald/models/lander-small/geometry/dAgpu0145_Fa_me --scalars /mnt/raid/wald/models/lander-small/10000unsteadyiters/dAgpu0145_Fa_volume_data. -o /space/lander-vopat-animated/lander-small-rho-7000.umesh -var rho -ts 7000
~/Projects/vopat/bin/vopatPartitionUMeshSpatially /space/lander-vopat-animated/lander-small-rho-7000.umesh -n 8 -o /space/lander-vopat-animated/lander-small-rho-7000-n8-spatial.umesh
~/Projects/vopat/bin/vopatPartitionUMeshObjectSpace /space/lander-vopat-animated/lander-small-rho-7000.umesh -n 8 -o /space/lander-vopat-animated/lander-small-rho-7000-n8-object.umesh

for f in 0 1 2 3 4 5 6 7; do
    ./fun3DExtractVariable --umesh /space/lander-vopat-animated/lander-small-rho-7000-n8-spatial.umesh.b0000$f.unvar.brick.umesh --volume-data /mnt/raid/wald/models/lander-small/10000unsteadyiters/dAgpu0145_Fa_volume_data. -var rho -o /space/lander-vopat-animated/lander-small-rho-7000-n8-spatial.umesh.b0000$f.unvar.brick.scalars_rho
    ./fun3DExtractVariable --umesh /space/lander-vopat-animated/lander-small-rho-7000-n8-object.umesh.b0000$f.unvar.brick.umesh --volume-data /mnt/raid/wald/models/lander-small/10000unsteadyiters/dAgpu0145_Fa_volume_data. -var rho -o /space/lander-vopat-animated/lander-small-rho-7000-n8-object.umesh.b0000$f.unvar.brick.scalars_rho
done

