

#ifndef FULLBOARD_H_INCLUDED
#define FULLBOARD_H_INCLUDED

#include "config.h"
#include <cstdint>
#include "FastBoard.h"

#include "pattern3x3.h"
#include "distance.h"

////////////////////////////////////////////////////////////////////////
typedef unsigned long long int int64;

#ifdef _WIN32
	#define COMPILER_MSVC
#endif

#if defined (_M_IX86)
inline int popcnt32twice(int64 bb) {
		return	_mm_popcnt_u32(unsigned int(bb)) + \
				_mm_popcnt_u32(unsigned int(bb >> 32));
	}
	#define popcnt64 popcnt32twice
#else
	#define popcnt64 _mm_popcnt_u64
#endif  /* defined (_M_IX86) */

/**
 *  �����Ȃ�64bit������NTZ�i�E���瑱��0�̌��j��߂�֐�
 *  Function for finding an unsigned 64-integer NTZ.
 *  Ex. 0x01011000 -> 3
 *      0x0 -> 64
 */
constexpr auto magic64 = 0x03F0A933ADCBD8D1ULL;
constexpr int ntz_table64[127] = {
    64,  0, -1,  1, -1, 12, -1,  2, 60, -1, 13, -1, -1, 53, -1,  3,
    61, -1, -1, 21, -1, 14, -1, 42, -1, 24, 54, -1, -1, 28, -1,  4,
    62, -1, 58, -1, 19, -1, 22, -1, -1, 17, 15, -1, -1, 33, -1, 43,
    -1, 50, -1, 25, 55, -1, -1, 35, -1, 38, 29, -1, -1, 45, -1,  5,
    63, -1, 11, -1, 59, -1, 52, -1, -1, 20, -1, 41, 23, -1, 27, -1,
    -1, 57, 18, -1, 16, -1, 32, -1, 49, -1, -1, 34, 37, -1, 44, -1,
    -1, 10, -1, 51, -1, 40, -1, 26, 56, -1, -1, 31, 48, -1, 36, -1,
     9, -1, 39, -1, -1, 30, 47, -1,  8, -1, -1, 46,  7, -1,  6,
};

inline constexpr int NTZ(int64 x) noexcept {
    return ntz_table64[static_cast<int64>(magic64*static_cast<int64>(x&-x))>>57];
}

extern const double response_w[4][2];
extern const double semeai_w[2][2];

//#define USE_52FEATURE

/**************************************************************
 *
 *  �u�A�v�̃N���X
 *
 *  �אڂ��Ă���Γ��m�͈�̘A��`������.
 *  �A�ɗאڂ����_��u�ċz�_�v�ƌĂсA
 *  �ċz�_��0�ɂȂ�Ƃ��̘A�͎����.
 *
 *  �ċz�_�̍��W�͎��Ֆ�(19x19)�̃r�b�g�{�[�h(64bit����x6)�ŕێ����A
 *  �אڂ�����W�Ő΂��u�����/�������ꍇ�Ɍċz�_�𑝌�����.
 *
 *
 *  Class of Ren.
 *
 *  Adjacent stones form a single stone string (=Ren).
 *  Neighboring empty vertexes are call as 'liberty',
 *  and the Ren is captured when all liberty is filled.
 *
 *  Positions of the liberty are held in a bitboard
 *  (64-bit integer x 6), which covers the real board (19x19).
 *  The liberty number decrease (increase) when a stone is
 *  placed on (removed from) a neighboring position.
 *
 ***************************************************************/
class Ren {
public:

	// �ċz�_���W�̃r�b�g�{�[�h
	// Bitboard of liberty positions.
	// [0] -> 0-63, [1] -> 64-127, ..., [5] -> 320-360
	int64 lib_bits[6];

	int lib_atr; 	// �A�^���̏ꍇ�̌ċz�_���W. The liberty position in case of Atari.
	int lib_cnt;	// �ċz�_��. Number of liberty.
	int size;		// �A��\������ΐ�. Number of stones.

	Ren(){ Clear(); }
	Ren(const Ren& other){ *this = other; }
	Ren& operator=(const Ren& other){
		lib_cnt = other.lib_cnt;
		size 	= other.size;
		lib_atr = other.lib_atr;
		std::memcpy(lib_bits, other.lib_bits, sizeof(lib_bits));

		return *this;
	}

	// �΂̂���n�_�̏�����
	// Initialize for stone.
	void Clear(){
		lib_cnt 	= 0;
		size 		= 1;
		lib_atr 	= VNULL;
		lib_bits[0] = lib_bits[1] = lib_bits[2] = \
		lib_bits[3] = lib_bits[4] = lib_bits[5] = 0;
	}

	// �΂̂Ȃ��n�_�̏������i�����l�j
	// Initialize for empty and outer boundary.
	void SetNull(){
		lib_cnt 	= VNULL; //442
		size 		= VNULL;
		lib_atr 	= VNULL;
		lib_bits[0] = lib_bits[1] = lib_bits[2] = \
		lib_bits[3] = lib_bits[4] = lib_bits[5] = 0;
	}

	// ���Wv�Ɍċz�_��ǉ�����
	// Add liberty at v.
	void AddLib(int v){
		if(size == VNULL) return;

		int bit_idx = etor[v] / 64;
		int64 bit_v = (0x1ULL << (etor[v] % 64));

		if(lib_bits[bit_idx] & bit_v) return;
		lib_bits[bit_idx] |= bit_v;
		++lib_cnt;
		lib_atr = v;
	}

	// ���Wv�̌ċz�_���������
	// Delete liberty at v.
	void SubLib(int v){
		if(size == VNULL) return;

		int bit_idx = etor[v] / 64;
		int64 bit_v = (0x1ULL << (etor[v] % 64));
		if(lib_bits[bit_idx] & bit_v){
			lib_bits[bit_idx] ^= bit_v;
			--lib_cnt;

			if(lib_cnt == 1){
				for(int i=0;i<6;++i){
					if(lib_bits[i] != 0){
						lib_atr = rtoe[NTZ(lib_bits[i]) + i * 64];
					}
				}
			}
		}
	}

	// �ʂ̘Aother�ƘA������
	// Merge with another Ren.
	void Merge(const Ren& other){
		lib_cnt = 0;
		for(int i=0;i<6;++i){
			lib_bits[i] |= other.lib_bits[i];
			if(lib_bits[i] != 0){
				lib_cnt += (int)popcnt64(lib_bits[i]);
			}
		}
		if(lib_cnt == 1){
			for(int i=0;i<6;++i){
				if(lib_bits[i] != 0){
					lib_atr = rtoe[NTZ(lib_bits[i]) + i * 64];
				}
			}
		}
		size += other.size;
	}

	// �ċz�_����0��
	// Return whether this Ren is captured.
	bool IsCaptured() const{ return lib_cnt == 0; }

	// �ċz�_����1��
	// Return whether this Ren is Atari.
	bool IsAtari() const{ return lib_cnt == 1; }

	// �ċz�_����2��
	// Return whether this Ren is pre-Atari.
	bool IsPreAtari() const{ return lib_cnt == 2; }

	// �A�^���̘A�̌ċz�_��Ԃ�
	// Return the liberty position of Ren in Atari.
	int GetAtari() const{ return lib_atr; }

};
/////////////////////////////////////////////////////////////////////


class FullBoard : public FastBoard {
public:
    int remove_string(int i);
    int update_board(const int color, const int i);

    std::uint64_t calc_hash(int komove = 0);
    std::uint64_t calc_ko_hash(void);
    std::uint64_t get_hash(void) const;
    std::uint64_t get_ko_hash(void) const;
    void set_to_move(int tomove);

    void reset_board(int size);
    void display_board(int lastmove = -1);

    std::uint64_t m_hash;
    std::uint64_t m_ko_hash;

    
/////////////////////////////////////////////////////////////////////////////////
    
	bool IsSelfAtariNakade(int v) const;
    bool IsEyeShape(int pl, int v) const;//*****
    bool IsFalseEye(int v) const;//*****
    bool IsSeki(int v);
    bool IsLegal(int pl, int v) const;
    bool Semeai2(std::vector<int>& patr_rens, std::vector<int>& her_libs);
	bool Semeai3(std::vector<int>& lib3_rens, std::vector<int>& her_libs);
    int SelectMove();
    int SelectRandomMove();
    inline void SubProbDist();
    inline void SubPrevPtn();
    inline void AddProb(int pl, int v, double add_prob);
    inline void UpdatePrevPtn(int v);
    inline void PlaceStone(int v) ;
    inline void MergeRen(int v_base, int v_add) ;
    inline void RemoveStone(int v) ;
    inline void AddUpdatedPtn(int v);
    inline void RemoveRen(int v);
    inline void CancelAtari(int v);
    inline void CancelPreAtari(int v);
    inline void SetPreAtari(int v);
    inline bool IsSelfAtari(int pl, int v) const;
    inline void UpdateProbAll();
    inline void AddProbDist(int v);
    inline void SetAtari(int v); 
	void RecalcProbAll();
	void AddProbPtn12();
    void UpdateAQParameter(int v);
    void AQClear();
    void PlayLegal(int v); 
    void ReplaceProb(int pl, int v, double new_prob);

    int my,her;
    int color[EBVCNT];
    int prev_color[7][EBVCNT];
    int empty[BVCNT];
    int empty_idx[EBVCNT];
    int stone_cnt[2];
    int empty_cnt;
    int ko;
    int ren_idx[EBVCNT];
    int next_ren_v[EBVCNT];
    int move_cnt;
    bool is_ptn_updated[MAXSQ];
    double prev_ptn_prob;
    double sum_prob_rank[2][BSIZE];
    int prev_ko;
	Ren ren[MAXSQ];
    int pass_cnt[2];
    double prob[2][EBVCNT];
	Pattern3x3 prev_ptn[2];
	Pattern3x3 ptn[EBVCNT];
	int prev_move[2];
	int response_move[4];
    std::vector<int> semeai_move[2];
    std::vector<int> move_history;
    std::vector<int> removed_stones;
    std::vector<std::pair<int,int>> updated_ptns;

/////////////////////////////////////////////////////////////////////////////////

};


#endif
