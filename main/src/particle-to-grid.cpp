#include "p2g/config.hpp"
#include "p2g/run.hpp"
#include "p2g/utils.hpp"
#include "arg_parser.hpp"

#include <iostream>

using namespace sphexa;

namespace {

void printHelp(const char* name)
{
    printf("\nUsage:\n\n");
    printf("%s [OPTIONS]\n\n", name);
    printf("Options:\n");
    printf("  --checkpoint <path>     Input file (required)\n");
    printf("  --checkpoint-type <t>   'hdf5' (default) or 'tipsy'\n");
    printf("  --stepNo <n>            HDF5 step (default 0)\n");
    printf("  --lbox <L>              Box size; required for TIPSY\n");
    printf("  --rhoCrit <rho>         Critical density for TIPSY mass scaling\n");
    printf("  --gridSize <n>          Mesh dimension (default: 2^ceil(log2(cbrt(N))))\n");
    printf("  --field <name>[,name]   Extra particle fields to rasterize (e.g. temp,vx)\n");
    printf("  --interpolation <m>     'ngp' (default) or 'sph'\n");
    printf("  --maxSupportCells <n>   SPH stencil half-width cap, in cells (default 8)\n");
    printf("  --output-format <fmt>   'text' (default) or 'hdf5'\n");
    printf("  --output <path>         Output base name (default: 'density')\n");
    printf("  --no-output             Do not write output files\n");
    printf("  -h, --help              This help\n\n");
    printf("Mass is summed per cell and divided by the cell volume, giving a mass density that\n");
    printf("conserves total mass. Extra fields are averaged over the particles in the cell.\n\n");
    printf("Interpolation:\n");
    printf("  ngp  each particle lands in the single cell containing it.\n");
    printf("  sph  each particle is spread over the cells within its kernel support (2h), using\n");
    printf("       the cubic spline and the 'h' field from the checkpoint. Needs h to be present\n");
    printf("       and larger than a cell to differ from ngp; both conserve total mass.\n\n");
}

p2g::Config parseConfig(const ArgParser& parser)
{
    p2g::Config config;
    config.checkpoint_path   = parser.get("--checkpoint");
    config.checkpoint_type   = p2g::parseCheckpointType(parser.get("--checkpoint-type", std::string("hdf5")));
    config.step_no           = parser.get("--stepNo", 0);
    config.grid_size         = parser.get("--gridSize", 0);
    config.lbox              = parser.get("--lbox", 0.0);
    config.rho_crit          = parser.get("--rhoCrit", 0.0);
    config.extra_field_names = parser.getCommaList("--field");
    config.interpolation     = p2g::parseInterpolation(parser.get("--interpolation", std::string("ngp")));
    config.max_support_cells = parser.get("--maxSupportCells", 8);
    config.write_output      = !parser.exists("--no-output");
    config.output_path       = parser.get("--output", std::string("density"));
    config.output_format     = p2g::parseOutputFormat(parser.get("--output-format", std::string("text")));
    p2g::validate(config);
    return config;
}

} // namespace

int main(int argc, char** argv)
{
    auto [rank, numRanks] = p2g::initMpi();
    const ArgParser parser(argc, (const char**)argv);

    if (parser.exists("-h") || parser.exists("--help"))
    {
        if (rank == 0) printHelp(argv[0]);
        return p2g::exitSuccess();
    }

    p2g::Config config;
    try
    {
        config = parseConfig(parser);
    }
    catch (const std::exception& e)
    {
        if (rank == 0) std::cerr << e.what() << std::endl;
        return p2g::exitFailure();
    }

    try
    {
        p2g::run(config, rank, numRanks);
    }
    catch (const std::exception& e)
    {
        if (rank == 0)
            std::cerr << "Error: " << e.what() << "\nCheck --checkpoint path and --checkpoint-type.\n";
        return p2g::exitFailure();
    }

    return p2g::exitSuccess();
}
