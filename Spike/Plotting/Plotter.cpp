#include "Plotter.hpp"
#include "Spike/Helpers/TerminalHelpers.hpp"

#include <vector>
#include <string>

using namespace std;
// Plotter Constructor
Plotter::Plotter(string experimentName_param) {
	RESULTS_DIRECTORY = "output/"+experimentName_param+"/";

}


void Plotter::plot_single_cell_information_analysis(SpikeAnalyser * spike_analyser_for_untrained_network, SpikeAnalyser * spike_analyser_for_trained_network) {

	if (spike_analyser_for_untrained_network->descending_maximum_information_score_for_each_neuron == nullptr)
		return;
	if (spike_analyser_for_trained_network->descending_maximum_information_score_for_each_neuron == nullptr)
		return;
	if (spike_analyser_for_untrained_network->descending_maximum_information_score_for_each_neuron_vec.size() != spike_analyser_for_trained_network->descending_maximum_information_score_for_each_neuron_vec.size())
		return;

	mglGraph *gr = new mglGraph();
	gr->SetFontSize(2.0);
	gr->SetSize(8000,2000);

	int number_of_layers = spike_analyser_for_untrained_network->descending_maximum_information_score_for_each_neuron_vec.size();
	float maximum_possible_information_score = spike_analyser_for_untrained_network->maximum_possible_information_score;

	for (int l=0; l<number_of_layers;l++){
		int number_of_neurons_in_single_cell_analysis_group = spike_analyser_for_untrained_network->number_of_neurons_in_single_cell_analysis_group_vec[l];

		//plot max
		mglData dataUNTRAINED(number_of_neurons_in_single_cell_analysis_group);
		mglData dataTRAINED(number_of_neurons_in_single_cell_analysis_group);

		for (int neuron_id = 0; neuron_id < number_of_neurons_in_single_cell_analysis_group; neuron_id++) {

			dataUNTRAINED.a[neuron_id] = spike_analyser_for_untrained_network->descending_maximum_information_score_for_each_neuron_vec[l][neuron_id];
			dataTRAINED.a[neuron_id] = spike_analyser_for_trained_network->descending_maximum_information_score_for_each_neuron_vec[l][neuron_id];
		}

		gr->SubPlot(number_of_layers,2,l,"<_");
		gr->Title("Single Cell Information Analysis (Max)");
		//gr->Aspect(0.8,0.8,0.8);
		gr->SetRanges(0,number_of_neurons_in_single_cell_analysis_group,maximum_possible_information_score*-0.1,maximum_possible_information_score*1.2);
		gr->Axis();
		gr->Label('y',"Information [bit]",0);

		if (l==0){
			gr->Plot(dataUNTRAINED, "k|", "legend 'dataUNTRAINED'");
			gr->Plot(dataTRAINED, "k-", "legend 'dataTRAINED'");
			gr->Legend();
		}else{
			gr->Plot(dataUNTRAINED, "k|");
			gr->Plot(dataTRAINED, "k-");
		}







		//plot avg
		for (int neuron_id = 0; neuron_id < number_of_neurons_in_single_cell_analysis_group; neuron_id++) {

			dataUNTRAINED.a[neuron_id] = spike_analyser_for_untrained_network->descending_average_information_score_for_each_neuron_vec[l][neuron_id];
			dataTRAINED.a[neuron_id] = spike_analyser_for_trained_network->descending_average_information_score_for_each_neuron_vec[l][neuron_id];
		}

		gr->SubPlot(number_of_layers,2,number_of_layers+l,"<_");
		gr->Title("Single Cell Information Analysis (Average)");
		//gr->Aspect(0.8,0.8,0.8);
		gr->SetRanges(0,number_of_neurons_in_single_cell_analysis_group,maximum_possible_information_score*-0.1,maximum_possible_information_score*1.2);
		gr->Axis();
		gr->Label('y',"Information [bit]",0);

		gr->Plot(dataUNTRAINED, "k|");
		gr->Plot(dataTRAINED, "k-");









	}

	gr->WriteFrame((RESULTS_DIRECTORY+"single_cell_information_analysis.png").c_str());
	gr->WriteFrame((RESULTS_DIRECTORY+"single_cell_information_analysis.eps").c_str());





	printf("plot_single_cell_information_analysis implementation currently incomplete...\n");
}


void Plotter::multiple_subplots_test() {

	mglGraph *gr = new mglGraph();		// class for plot drawing

	gr->SetSize(1000,1000);

	gr->SubPlot(2,2,0,"<_");
	gr->Title("Semi-log axis");	gr->SetRanges(0.01,100,-1,1);	gr->SetFunc("lg(x)","");
	gr->Axis();	gr->Grid("xy","g");	gr->FPlot("sin(1/x)");	gr->Label('x',"x",0); gr->Label('y', "y = sin 1/x",0);
	gr->SubPlot(2,2,1,"<_");	gr->Title("Log-log axis");	gr->SetRanges(0.01,100,0.1,100);	gr->SetFunc("lg(x)","lg(y)");
	gr->Axis();	gr->Grid("!","h=");	gr->Grid();	gr->FPlot("sqrt(1+x^2)");	gr->Label('x',"x",0); gr->Label('y', "y = \\sqrt{1+x^2}",0);
	gr->SubPlot(2,2,2,"<_");	gr->Title("Minus-log axis");	gr->SetRanges(-100,-0.01,-100,-0.1);	gr->SetFunc("-lg(-x)","-lg(-y)");
	gr->Axis();	gr->FPlot("-sqrt(1+x^2)");	gr->Label('x',"x",0); gr->Label('y', "y = -\\sqrt{1+x^2}",0);
	gr->SubPlot(2,2,3,"<_");	gr->Title("Log-ticks");	gr->SetRanges(0.1,100,0,100);	gr->SetFunc("sqrt(x)","");
	gr->Axis();	gr->FPlot("x");	gr->Label('x',"x",1); gr->Label('y', "y = x",0);

	gr->WriteFrame("test_plot.png");	// save it



}
