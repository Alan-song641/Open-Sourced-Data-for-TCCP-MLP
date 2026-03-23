//-----------------------------------------------------------------------------------
// eOn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// A copy of the GNU General Public License is available at
// http://www.gnu.org/licenses/
//-----------------------------------------------------------------------------------

#ifndef PYAMFF_POTENTIAL
#define PYAMFF_POTENTIAL
#include <string>
#include <map>
#include <vector>
using namespace std;

extern "C" void calc_lammps(long *nAtoms, const double [], const double [], const int [], double [], double *U, int *num_elements, int [],int *MAX_FPS,int *MAX_NEIGHS,const int [],long *tAtoms, const int [], const double [], const int []);

extern "C" void read_mlffParas(long *nAtoms, int *num_elements, int *max_fps, const int [], int []);

extern "C" void cleanup();

extern "C" void nncleanup();

extern "C" void prepfNN( long *nAtoms, int *num_elements, int *max_fps, const int [], int []);

class PyAMFFInterface
{

    public:
        PyAMFFInterface(void);
		~PyAMFFInterface();
        void updateTotalAtoms(long tatoms);
		void initialize(int max_fps,long natoms, int ntypes,int* types,double* mass, double ** pos_car, double dbox[3][3], double bound[3][2], bool ortho);
		void cleanMemory(void);
        void process();
        void setMaxNeighs(int max);
        void updatePosition(int i,int tag, double* pos);
        void addNeighbor(int i, int j);
        double getForcesEnergy(double* const* const& forces);
        double getMaxCutoff() const;
        struct element  {
            std::string symbol;
            double mass;
            int number;
        };
        bool isNew() const;
        std::string getElementSymbol(int number);
        std::string getElementSymbol(double mass);
        std::string getElementSymbol(double mass, double tol);
        int getElementNumber(double mass);
        int getElementNumber(double mass,double tol);
        int getElementNumber(std::string symbol);
        double getElementMass(int number);
        double getElementMass(std::string symbol);
    protected:
        bool new_pyamff;

    private:
        void reset(); //find a way beyond this
        void setup();
        bool noNeighs, is_orthogonal;
        long num_atoms, total_atoms;
        double energy;
        double box[9], super[9];
        double *positions, *forces, *pool_positions;
        int num_cells[3];
        map<string,vector<double>> orig_bounds;
        map<string,vector<double>> bounds;
        int max_fps, num_types, max_neighs;//, total_pairs, max_neighs;
        int *atomicNrs, *unique, *num_neighs, *neigh_tag;//, *num_pair;
        vector<vector<int>> neighs, adjust;
        // vector<vector<double>> distances;
        struct element PeriodicTable[119] = {//NOTE TO SELF: MAKE THIS A VECTOR OR A HASH MAP FOR BETTER SPEED
            {  "H",     1.00794000,  1},
            { "He",       4.00260200,  2},
            { "Li",      6.94100000,  3},
            { "Be",    9.01218200,  4},
            {  "B",       10.81100000,  5},
            {  "C",      12.01070000,  6},
            {  "N",    14.00670000,  7},
            {  "O",      15.99940000,  8},
            {  "F",    18.99840320,  9},
            { "Ne",        20.17970000,  10},
            { "Na",      22.98976928,  11},
            { "Mg",   24.30500000,  12},
            { "Al",    26.98153860,  13},
            { "Si",     28.08550000,  14},
            {  "P",  30.97376200,  15},
            {  "S",      32.06500000,  16},
            { "Cl",    35.45300000,  17},
            { "Ar",       39.94800000,  18},
            {  "K",   39.09830000,  19},
            { "Ca",     40.07800000,  20},
            { "Sc",    44.95591200,  21},
            { "Ti",    47.86700000,  22},
            {  "V",    50.94150000,  23},
            { "Cr",    51.99610000,  24},
            { "Mn",   54.93804500,  25},
            { "Fe",        55.84500000,  26},
            { "Co",      58.69340000,  27},
            { "Ni",      58.93319500,  28},
            { "Cu",      63.54600000,  29},
            { "Zn",        65.38000000,  30},
            { "Ga",     69.72300000,  31},
            { "Ge",   72.64000000,  32},
            { "As",     74.92160000,  33},
            { "Se",    78.96000000,  34},
            { "Br",     79.90400000,  35},
            { "Kr",     83.79800000,  36},
            { "Rb",    85.46780000,  37},
            { "Sr",   87.62000000,  38},
            {  "Y",     88.90585000,  39},
            { "Zr",   91.22400000,  40},
            { "Nb",     92.90638000,  41},
            { "Mo",  95.96000000,  42},
            { "Tc",  98.00000000,  43},
            { "Ru",  101.07000000,  44},
            { "Rh",    102.90550000,  45},
            { "Pd",  106.42000000,  46},
            { "Ag",     107.86820000,  47},
            { "Cd",    112.41100000,  48},
            { "In",     114.81800000,  49},
            { "Sn",        118.71000000,  50},
            { "Sb",   121.76000000,  51},
            { "Te",  127.60000000,  52},
            {  "I",     126.90470000,  53},
            { "Xe",      131.29300000,  54},
            { "Cs",     132.90545190,  55},
            { "Ba",     137.32700000,  56},
            { "La",  138.90547000,  57},
            { "Ce",     140.11600000,  58},
            { "Pr", 140.90765000,  59},
            { "Nd",  144.24200000,  60},
            { "Pm", 145.00000000,  61},
            { "Sm",   150.36000000,  62},
            { "Eu",   151.96400000,  63},
            { "Gd", 157.25000000,  64},
            { "Tb",    158.92535000,  65},
            { "Dy", 162.50000000,  66},
            { "Ho",    164.93032000,  67},
            { "Er",     167.25900000,  68},
            { "Tm",    168.93421000,  69},
            { "Yb",  173.05400000,  70},
            { "Lu",   174.96680000,  71},
            { "Hf",    178.49000000,  72},
            { "Ta",   180.94788000,  73},
            {  "W",   183.84000000,  74},
            { "Re",    186.20700000,  75},
            { "Os",     190.23000000,  76},
            { "Ir",    192.21700000,  77},
            { "Pt",   195.08400000,  78},
            { "Au",       196.96656900,  79},
            { "Hg",    200.59000000,  80},
            { "Tl",   204.38330000,  81},
            { "Pb",       207.20000000,  82},
            { "Bi",    208.98040000,  83},
            { "Po",   209.00000000,  84},
            { "At",   210.00000000,  85},
            { "Rn",      220.00000000,  86},
            { "Fr",   223.00000000,  87},
            { "Ra",     226.00000000,  88},
            { "Ac",   227.00000000,  89},
            { "Th",    231.03588000,  90},
            { "Pa", 232.03806000,  91},
            {  "U",    237.00000000,  92},
            { "Np",  238.02891000,  93},
            { "Pu",  243.00000000,  94},
            { "Am",  244.00000000,  95},
            { "Cm",     247.00000000,  96},
            { "Bk",  247.00000000,  97},
            { "Cf",  251.00000000,  98},
            { "Es",  252.00000000,  99},
            { "Fm",    257.00000000,  100},
            { "Md",  258.00000000,  101},
            { "No",   259.00000000,  102},
            { "Lr", 266.00000000,  103},
            { "Rf", 267.00000000,  104},
            { "Db",    268.00000000,  105},
            { "Sg", 269.00000000,  106},
            { "Bh",    270.00000000,  107},
            { "Hs",    277.00000000,  108},
            { "Mt", 278.00000000,  109},
            { "Ds", 281.00000000,  110},
            { "Rg",  282.00000000,  111},
            { "Cn",  285.00000000,  112},
            { "Nh",   286.00000000,  113},
            { "Fl",  289.00000000,  114},
            { "Mc",  290.00000000,  115},
            { "Lv",  293.00000000,  116},
            { "Ts", 294.00000000,  117},
            { "Og",  294.00000000,  118},
            { "Xx",      1.00000000,  0}
        };

        element getElement(int number);
        element getElement(double mass);
        element getElement(double mass, double tol);
        element getElement(std::string sym);
};     

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool PyAMFFInterface::isNew() const
{
    return new_pyamff;
}
#endif

