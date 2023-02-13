
#ifndef ZOBRIST_H_INCLUDED
#define ZOBRIST_H_INCLUDED

#include "config.h"

#include <array>
#include <cstdint>

#include "FastBoard.h"
#include "Random.h"

#include <random>
#include <unordered_map>
#include "config.h"

class Zobrist {
public:
    static constexpr auto zobrist_empty = 0x1234567887654321;
    static constexpr auto zobrist_blacktomove = 0xABCDABCDABCDABCD;

    static std::array<std::array<std::uint64_t, FastBoard::MAXSQ>,     4> zobrist;
    static std::array<std::uint64_t, FastBoard::MAXSQ>                    zobrist_ko;
    static std::array<std::array<std::uint64_t, FastBoard::MAXSQ * 2>, 2> zobrist_pris;
    static std::array<std::uint64_t, 5>                                   zobrist_pass;

    static void init_zobrist(Random& rng);
};

#endif


///////////////////////////////////////////////////////////////////////////////


typedef unsigned long long int int64;

extern std::mt19937 mt_32;
extern std::mt19937_64 mt_64;
extern std::uniform_real_distribution<double> mt_double;
extern std::uniform_int_distribution<int> mt_int8;


/**
 *  Zobrist�n�b�V���̃V�[�h��ێ�����e�[�u��
 *  Class containing the table for Zobrist hash.
 */
struct Zobrist1{

	int64 hash[8][4][EBVCNT];

	// axis=0: symmetrical index
	//         [0]->original [1]->90deg rotation, ... [4]->inverting, ...
	// axis=1: stone color index
	//         [0]->blank [1]->Ko [2]->white stone [3]->black stone
	// axis=2: vertex position

	Zobrist1(){

		std::mt19937_64 mt_64_(123);

		for (int i=0;i<8;++i){
			bool is_inverted = (i == 4);

			for (int j=0;j<4;++j) {
				for (int k=0;k<EBVCNT;++k) {

					int x = EBSIZE - 1 - etox[k];
					int y = etoy[k];

					if(i == 0)				hash[i][j][k] = mt_64_();
					else if(is_inverted)	hash[i][j][k] = hash[0][j][xytoe[x][y]];
					else					hash[i][j][k] = hash[i - 1][j][xytoe[y][x]];

				}
			}
		}

	}

}; const Zobrist1 zobrist1;


/**
 *  �i�J�f�̋}����i�[����N���X
 *     key: ��_��Zobrist�n�b�V��
 *     value: �}���̍��W
 *  ���W�͔Ւ�������̑��΍��W.
 *
 *  Class containing Nakade patterns and their vitals.
 *     key: Zobrist hash of blank vertexes.
 *     value: position of vital
 *
 *  Each positions are described by relative coordinate
 *  with the origin at the board center.
 *  Ex. v_rel = v - EBVCNT / 2
 */
struct Nakade{

	std::unordered_map<int64,int> vital;

	Nakade(){

		int space_3[4][3] = {	{-1,0,1},			// 0 <- vital position
								{0,1,EBSIZE},		// 0
								{0,1,2},			// 1
								{0,1,EBSIZE+1}	};	// 1

		int space_4[4][4] = {	{-1,0,1,EBSIZE},			// 0
								{0,1,-EBSIZE+1,EBSIZE+1},	// 1
								{0,1,2,EBSIZE+1},			// 1
								{0,1,EBSIZE,EBSIZE+1}};		// 0

		int space_5[7][5] = {	{-1,0,1,EBSIZE-1,EBSIZE},			// 0
								{-1,0,1,-EBSIZE,EBSIZE},			// 0
								{0,1,2,EBSIZE,EBSIZE+1},			// 1
								{0,1,2,EBSIZE+1,EBSIZE+2},			// 1
								{0,1,-EBSIZE+1,EBSIZE,EBSIZE+1},	// 1
								{0,1,2,-EBSIZE+1,EBSIZE+1},			// 1
								{0,1,EBSIZE,EBSIZE+1,EBSIZE+2}};	// EBSIZE+1

		int space_6[4][6] = {	{-EBSIZE-1,-EBSIZE,-1,0,1,EBSIZE},			// 0
								{0,1,2,-EBSIZE+1,EBSIZE,EBSIZE+1},			// 1
								{-EBSIZE+1,0,1,2,EBSIZE+1,EBSIZE+2},		// 1
								{0,1,EBSIZE,EBSIZE+1,EBSIZE+2,2*EBSIZE+1}};	// EBSIZE+1

		int vaital_3[4] = {0,0,1,1};
		int vaital_4[4] = {0,1,1,0};
		int vaital_5[7] = {0,0,1,1,1,1,2};
		int vaital_6[4] = {0,1,1,2};

		int sym_position[4][8] = {	{0,0,0,0,0,0,0,0},
									{1,-EBSIZE,-1,EBSIZE,-1,-EBSIZE,1,EBSIZE},
									{EBSIZE+1,-EBSIZE+1,-EBSIZE-1,EBSIZE-1,EBSIZE-1,-EBSIZE-1,-EBSIZE+1,EBSIZE+1},
									{VNULL,VNULL,VNULL,VNULL,VNULL,VNULL,VNULL,VNULL}	};

		int center = EBVCNT / 2;
		int64 space_hash = 0;

		for(int i=0;i<8;++i){
			for(int j=0;j<4;++j){
				space_hash = 0;
				for(int k=0;k<3;++k){
					space_hash ^= zobrist1.hash[i][0][center+space_3[j][k]];
				}
				vital.insert({space_hash, sym_position[vaital_3[j]][i]});
			}
			for(int j=0;j<4;++j){
				space_hash = 0;
				for(int k=0;k<4;++k){
					space_hash ^= zobrist1.hash[i][0][center+space_4[j][k]];
				}
				vital.insert({space_hash, sym_position[vaital_4[j]][i]});
			}
			for(int j=0;j<7;++j){
				space_hash = 0;
				for(int k=0;k<5;++k){
					space_hash ^= zobrist1.hash[i][0][center+space_5[j][k]];
				}
				vital.insert({space_hash, sym_position[vaital_5[j]][i]});
			}
			for(int j=0;j<4;++j){
				space_hash = 0;
				for(int k=0;k<6;++k){
					space_hash ^= zobrist1.hash[i][0][center+space_6[j][k]];
				}
				vital.insert({space_hash, sym_position[vaital_6[j]][i]});
			}
		}

	}

}; const Nakade nakade;

