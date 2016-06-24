#!/usr/bin/perl

	#
	#  ParamGenAndRunVisNet.pl
	#  VisBack
	#
	#  Created by Akihiro Eguchi on 15/12/15.
	#  Copyright 2015 OFTNAI. All rights reserved.
	#
	#  This is a perl script to be called by dakota optimizer

	use strict;
    use warnings;
    use POSIX;
	use File::Copy;
	use File::Copy 'cp';
	use Data::Dumper;
	#use Data::Compare;
	use Cwd 'abs_path';
	
	### LOADING INPUT ###
	my ($infi, $outdir, $simulation_index) = @ARGV;
	
	open(my $fh, "<:encoding(UTF-8)", $infi)
		or die "Could not open";
	
	my @table;
	while (my $row = <$fh>){
		chomp $row;
		my $white = substr($row, 0, 1);
		my @rowInput;
		my $flag = 0;
		my $tmpStr = '';
		for (my $i=0;$i<length($row);$i++){
			my $char = substr($row, $i, 1);
			#print "$char\n";
			if ($char ne $white){
				$flag = 1;
				$tmpStr = $tmpStr . $char;
				#print($tmpStr);
				#print($char);
			}elsif($flag == 1){
				#print("$tmpStr\n");
				push(@rowInput,$tmpStr);
				$tmpStr = '';
				$flag = 0;
			}
			if($i==length($row)-1){
				#print("$tmpStr\n");
				push(@rowInput,$tmpStr);
				$tmpStr = '';
				$flag = 0;
			}
		}
	
	#	print("@rowInput\n");
		push(@table,[@rowInput]);	
	}
	close $fh;
	my $par1 = sprintf("%.11f", $table[1][0]);
	my $par2 = sprintf("%.11f", $table[2][0]);
	my $par3 = sprintf("%.11f", $table[3][0]);
	my $par4 = sprintf("%.11f", $table[4][0]);
	# my $par5 = sprintf("%.11f", $table[5][0]);
	# my $par6 = sprintf("%.11f", $table[6][0]);
	# my $par7 = sprintf("%.11f", $table[7][0]);
	# my $par8 = sprintf("%.11f", $table[8][0]);
	# print($table[1][0]);

	
#	system("python ../../dakota_runMe.py $outdir $par1 $par2 $par3 $par4 $par5");
	# system("python ../JI_Empty_Test.py $outdir $simulation_index $par1 $par2 $par3 $par4 $par5 $simulation_index");
	# system("../../Experiments/bin/ConductanceExperiment1 $outdir $simulation_index $par1 $par2 $par3 $par4 $par5 $par6 $par7 $par8");
	system("../../Experiments/bin/ConductanceExperiment1 $outdir $simulation_index 2 $par1 $par2 $par3 $par4");
	
