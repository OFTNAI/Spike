#!/usr/bin/perl

	#
	#  ParamGenAndRunVisNet.pl
	#  VisBack
	#
	#  Created by Akihiro Eguchi on 15/12/15.
	#  Copyright 2015 OFTNAI. All rights reserved.
	#
	#  This is a perl script to be called by dakota optimizer

	# Adapted by James Isbister for use with Spike

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


	# Read optimisation iteration index from file
	my $optimisation_iteration_index_filename = '../optimisation_iteration_index.txt';
	open(my $optimisation_iteration_index_text, '<:encoding(UTF-8)', $optimisation_iteration_index_filename)
  		or die "Could not open file '$optimisation_iteration_index_filename' $!";
 
 	my $optimisation_iteration_index;
	while (my $row = <$optimisation_iteration_index_text>) {
  		chomp $row;
  		# print "$row\n";
  		$optimisation_iteration_index = $row;
	}	


	# Use when optimising entire network layer by layer.
	system("../../Experiments/bin/ConductanceExperiment1 $outdir $simulation_index $optimisation_iteration_index $par1 $par2 $par3 $par4 1");

	# Use when optimising particular layer.
	# system("../../Experiments/bin/ConductanceExperiment1 $outdir $simulation_index 0 $par1 $par2 $par3 $par4 0");
