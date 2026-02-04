#include "p2g/config.hpp"
#include "p2g/interpolation_method.hpp"
#include "p2g/run.hpp"
#include "p2g/utils.hpp"
#include "arg_parser.hpp"

#include <cstring>
#include <iostream>

using namespace sphexa;

namespace {

void printHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n\n", name);
        printf("Options:\n");
        printf("  --checkpoint <path>     Input file (required)\n");
        printf("  --checkpoint-type <t>   'hdf5' (default) or 'tipsy'\n");
        printf("  --stepNo <n>            HDF5 step (default 0)\n");
        printf("  --lbox <L>              Box size; required for TIPSY\n");
        printf("  --rhoCrit <rho>         Critical density for TIPSY mass scaling\n");
        printf("  --gridSize <n>          Mesh dimension (default: 2^ceil(log3(N)))\n");
        printf("  --interpolation <m>     'nearest', 'sph', or 'cell_average' (default: nearest)\n");
        printf("  --field <name>[,name]   Extra particle fields to rasterize (e.g. temp,vx)\n");
        printf("  --output-format <fmt>   'text' (default) or 'hdf5'\n");
        printf("  --output <path>         Output base name (default: 'density'); ext added automatically\n");
        printf("  --no-output             Do not write output files\n");
        printf("  -h, --help              This help\n\n");
    }
}

bool parseConfig(const ArgParser& parser, int rank, p2g::Config& config)
{
    config.checkpoint_path = parser.get("--checkpoint");
    config.checkpoint_type = parser.get("--checkpoint-type", std::string("hdf5"));
    config.step_no         = parser.get("--stepNo", 0);
    config.grid_size       = parser.get("--gridSize", 0);
    config.lbox            = parser.get("--lbox", 0.0);
    config.rho_crit        = parser.get("--rhoCrit", 0.0);
    config.extra_field_names = parser.getCommaList("--field");
    config.write_output      = !parser.exists("--no-output");
    config.output_path       = parser.get("--output", std::string("density"));
    std::string interpStr    = parser.get("--interpolation", std::string("nearest"));
    std::string outputFmtStr = parser.get("--output-format", std::string("text"));
    try
    {
        config.interpolation  = p2g::parseInterpolationMethod(interpStr);
        config.output_format  = p2g::parseOutputFormat(outputFmtStr);
    }
    catch (const std::exception& e)
    {
        if (rank == 0) std::cerr << e.what() << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    auto [rank, numRanks] = p2g::initMpi();
    const ArgParser parser(argc, (const char**)argv);

    if (parser.exists("-h") || parser.exists("--help"))
    {
        printHelp(argv[0], rank);
        return p2g::exitSuccess();
    }

    p2g::Config config;
    if (!parseConfig(parser, rank, config))
        return p2g::exitSuccess();

    std::string errMsg;
    if (!p2g::validate(config, rank, errMsg))
    {
        if (rank == 0 && !errMsg.empty()) std::cerr << errMsg << std::endl;
        return p2g::exitSuccess();
    }

    try
    {
        if (!p2g::run(config, rank, numRanks))
            return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        if (rank == 0) std::cerr << "Error: " << e.what() << "\nCheck --checkpoint path and --checkpoint-type.\n";
        return EXIT_FAILURE;
    }

    return p2g::exitSuccess();
}
