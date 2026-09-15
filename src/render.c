/*  space-edit renderer
 *  ------------------------------------------------------------------
 *  Procedural, beat-driven visuals for "singularity.wav" (VUTRA).
 *  Three scenes, all raymarched / procedural - no stock footage:
 *     mode 0  Schwarzschild black hole  (geodesic integration of
 *             u'' = -u + 3/2 rs u^2 in the ray plane, accretion disk
 *             with differential rotation + relativistic beaming,
 *             gravitationally lensed starfield & nebula behind it)
 *     mode 1  hyperspace warp          (radially streaked starfield)
 *     mode 2  interior / singularity   (collapsing filament tunnel)
 *
 *  Per-frame camera + FX parameters are read from a CSV timeline that
 *  tools/plan_edit.py derives from analysis/beatmap.json, so every
 *  zoom punch, cut, flash and flare lands on an actual onset.
 *
 *  usage: render <timeline.csv> <W> <H> [firstFrame] [lastFrame]
 *         writes rgb24 frames to stdout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* small vector helpers                                                */
typedef struct { float x, y, z; } V3;
static inline V3 v3(float x, float y, float z){ V3 r={x,y,z}; return r; }
static inline V3 add(V3 a, V3 b){ return v3(a.x+b.x,a.y+b.y,a.z+b.z); }
static inline V3 sub(V3 a, V3 b){ return v3(a.x-b.x,a.y-b.y,a.z-b.z); }
static inline V3 mul(V3 a, float s){ return v3(a.x*s,a.y*s,a.z*s); }
static inline V3 vmul(V3 a, V3 b){ return v3(a.x*b.x,a.y*b.y,a.z*b.z); }
static inline float dot(V3 a, V3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline V3 cross(V3 a, V3 b){ return v3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
static inline float len(V3 a){ return sqrtf(dot(a,a)); }
static inline V3 norm(V3 a){ float l=len(a); return l>0?mul(a,1.f/l):a; }
static inline float clampf(float v, float a, float b){ return v<a?a:(v>b?b:v); }
static inline float mixf(float a, float b, float t){ return a+(b-a)*t; }
static inline V3 mixv(V3 a, V3 b, float t){ return v3(mixf(a.x,b.x,t),mixf(a.y,b.y,t),mixf(a.z,b.z,t)); }
static inline float smoothstepf(float e0, float e1, float x){
    float t = clampf((x-e0)/(e1-e0+1e-9f),0.f,1.f); return t*t*(3.f-2.f*t);
}

/* ------------------------------------------------------------------ */
/* hashing & noise                                                     */
static inline unsigned int uhash(unsigned int x){
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static inline float h1(unsigned int x){ return (float)(uhash(x) & 0xffffffU) / 16777216.f; }
static inline float h2i(int x, int y, int s){
    return h1((unsigned)(x*374761393 + y*668265263 + s*1274126177));
}
static inline float h3i(int x, int y, int z, int s){
    return h1((unsigned)(x*374761393 + y*668265263 + z*2147483647 + s*1274126177));
}
static float vnoise2(float x, float y, int s){
    float fx = floorf(x), fy = floorf(y);
    int ix = (int)fx, iy = (int)fy;
    float tx = x-fx, ty = y-fy;
    tx = tx*tx*(3-2*tx); ty = ty*ty*(3-2*ty);
    float a = h2i(ix,iy,s),   b = h2i(ix+1,iy,s);
    float c = h2i(ix,iy+1,s), d = h2i(ix+1,iy+1,s);
    return mixf(mixf(a,b,tx), mixf(c,d,tx), ty);
}
static float vnoise3(float x, float y, float z, int s){
    float fx=floorf(x), fy=floorf(y), fz=floorf(z);
    int ix=(int)fx, iy=(int)fy, iz=(int)fz;
    float tx=x-fx, ty=y-fy, tz=z-fz;
    tx=tx*tx*(3-2*tx); ty=ty*ty*(3-2*ty); tz=tz*tz*(3-2*tz);
    float c000=h3i(ix,iy,iz,s),     c100=h3i(ix+1,iy,iz,s);
    float c010=h3i(ix,iy+1,iz,s),   c110=h3i(ix+1,iy+1,iz,s);
    float c001=h3i(ix,iy,iz+1,s),   c101=h3i(ix+1,iy,iz+1,s);
    float c011=h3i(ix,iy+1,iz+1,s), c111=h3i(ix+1,iy+1,iz+1,s);
    float x00=mixf(c000,c100,tx), x10=mixf(c010,c110,tx);
    float x01=mixf(c001,c101,tx), x11=mixf(c011,c111,tx);
    return mixf(mixf(x00,x10,ty), mixf(x01,x11,ty), tz);
}
static float fbm3(V3 p, int oct, int s){
    float a=0.5f, f=1.f, sum=0.f, nrm=0.f;
    for(int i=0;i<oct;i++){
        sum += a*vnoise3(p.x*f,p.y*f,p.z*f,s+i*17);
        nrm += a; a*=0.5f; f*=2.03f;
    }
    return sum/nrm;
}
static float fbm2(float x, float y, int oct, int s){
    float a=0.5f, f=1.f, sum=0.f, nrm=0.f;
    for(int i=0;i<oct;i++){
        sum += a*vnoise2(x*f,y*f,s+i*31);
        nrm += a; a*=0.5f; f*=2.07f;
    }
    return sum/nrm;
}

/* ------------------------------------------------------------------ */
/* frame parameters                                                    */
enum { P_TIME, P_MODE, P_CX, P_CY, P_CZ, P_TX, P_TY, P_TZ, P_ROLL, P_FOV,
       P_SHX, P_SHY, P_ZOOM, P_FLASH, P_EXPO, P_CHROMA, P_RBLUR, P_DISK,
       P_JET, P_NEB, P_STAR, P_TWK, P_BLOOM, P_GRAIN, P_VIGN, P_WARP,
       P_TUNNEL, P_HUE, P_SAT, P_TEXT, P_TALPHA, P_TSCALE, P_TOFF, P_FADE,
       P_SEED, P_INVERT, P_CONTRAST, P_DISKTEMP, P_STARDENS, P_RINGS,
       P_GLITCH, P_MIRROR, P_SPIN, P_SHUTTER, P_LIGHTAZ, P_LIGHTEL,
       NPARAM };

typedef struct { float p[NPARAM]; } Frame;

/* text overlay images (raw RGBA) */
#define MAXTEX 12
typedef struct { int w,h; unsigned char *px; } Tex;
static Tex g_tex[MAXTEX];
static int g_ntex = 0;

/* ------------------------------------------------------------------ */
/* starfield: near-uniform cube-face grid, analytic AA                  */
static void cubeface(V3 d, int *face, float *u, float *v){
    float ax=fabsf(d.x), ay=fabsf(d.y), az=fabsf(d.z);
    if(ax>=ay && ax>=az){ *face = d.x>0?0:1; *u=(d.x>0?-d.z:d.z)/ax; *v=d.y/ax; }
    else if(ay>=az){      *face = d.y>0?2:3; *u=d.x/ay; *v=(d.y>0?d.z:-d.z)/ay; }
    else {                *face = d.z>0?4:5; *u=(d.z>0?d.x:-d.x)/az; *v=d.y/az; }
}
/* returns emitted star colour for direction d; pixAng = radians/pixel */
static V3 stars(V3 d, float pixAng, float t, float twinkle, float bright, float dens){
    int face; float u,v;
    cubeface(d,&face,&u,&v);
    V3 acc = v3(0,0,0);
    const int   N[3]   = {80, 170, 360};
    const float BR[3]  = {0.75f, 0.34f, 0.16f};
    const float PR[3]  = {0.40f, 0.40f, 0.38f};   /* fraction of cells with a star */
    for(int L=0;L<3;L++){
        float n = (float)N[L];
        float cu = (u*0.5f+0.5f)*n, cv = (v*0.5f+0.5f)*n;
        int ix=(int)floorf(cu), iy=(int)floorf(cv);
        int seed = face*7919 + L*104729;
        float r0 = h2i(ix,iy,seed);
        if(r0 > PR[L]*dens) continue;
        float rx = 0.15f + 0.70f*h2i(ix,iy,seed+1);
        float ry = 0.15f + 0.70f*h2i(ix,iy,seed+2);
        float du = (cu-ix-rx)*(2.f/n), dv = (cv-iy-ry)*(2.f/n);
        float ang = sqrtf(du*du+dv*dv)*0.7854f;          /* uv -> approx radians */
        float mag = h2i(ix,iy,seed+3);
        float b = powf(mag, 7.0f)*BR[L];
        float tw = 1.f + twinkle*0.9f*sinf(t*(1.7f+4.f*mag) + mag*53.f);
        b *= (tw>0?tw:0);
        float rad = pixAng*(1.05f + 0.85f*b);
        float g = expf(-(ang*ang)/(rad*rad+1e-12f));
        if(g < 0.0025f) continue;
        float ct = h2i(ix,iy,seed+4);                     /* colour temperature */
        V3 col = ct<0.55f ? mixv(v3(0.62f,0.78f,1.00f), v3(1.f,1.f,1.f), ct/0.55f)
                          : mixv(v3(1.f,0.96f,0.88f), v3(1.f,0.72f,0.48f), (ct-0.55f)/0.45f);
        float spike = 0.f;
        if(b > 0.35f){                                    /* subtle diffraction cross */
            float sx = expf(-fabsf(dv)*1400.f/pixAng*0.0006f) * expf(-fabsf(du)*90.f/pixAng*0.0006f);
            float sy = expf(-fabsf(du)*1400.f/pixAng*0.0006f) * expf(-fabsf(dv)*90.f/pixAng*0.0006f);
            spike = (sx+sy)*0.10f*b;
        }
        acc = add(acc, mul(col, (g*b*13.f + spike)*bright));
    }
    return acc;
}

/* deep-space nebula, sampled on the escape direction */
static V3 nebula(V3 d, float t, float amt, int seed){
    if(amt <= 0.001f) return v3(0,0,0);
    V3 p = mul(d, 2.15f);
    float w = fbm3(mul(p,2.3f), 3, seed+301);
    V3 q = add(p, mul(v3(w-0.5f, w*0.8f-0.4f, 0.55f-w), 1.35f));
    float f    = fbm3(q, 4, seed);
    float mask = fbm3(mul(p,0.42f), 3, seed+77);
    /* large voids: most of the sky stays black */
    float region = powf(clampf((mask-0.43f)*3.0f,0.f,1.f), 1.5f);
    float dens = powf(clampf((f-0.47f)*2.6f,0.f,1.f), 2.1f) * region;
    float fil  = powf(clampf((fbm3(mul(q,4.6f),3,seed+909)-0.56f)*3.6f,0.f,1.f), 2.2f) * region;
    V3 cool = v3(0.06f,0.13f,0.40f);
    V3 warm = v3(0.52f,0.11f,0.46f);
    V3 hot  = v3(0.95f,0.42f,0.70f);
    V3 cyan = v3(0.10f,0.66f,0.82f);
    float mixr = clampf(fbm3(mul(p,0.31f),2,seed+512)*1.8f-0.35f,0.f,1.f);
    V3 col = mixv(cool, warm, mixr);
    col = mixv(col, hot, powf(dens,2.6f)*0.8f);
    col = mixv(col, cyan, fil*0.45f);
    return mul(add(mul(col, dens*1.30f), mul(cyan, fil*0.07f)), amt);
}

/* accretion disk emission at world point X (disk plane y = 0, rs = 1) */
static V3 disk_sample(V3 X, float t, float bright, float tempMix, float *alpha, int seed){
    float r = sqrtf(X.x*X.x + X.z*X.z);
    const float rin = 2.6f, rout = 14.0f;
    if(r < rin*0.9f || r > rout){ *alpha = 0.f; return v3(0,0,0); }
    float ang = atan2f(X.z, X.x);
    /* mild differential rotation: enough shear for spiral arms, not hair */
    float a2 = ang - t*(0.085f + 0.62f/powf(r,1.5f));
    float ca = cosf(a2), sa = sinf(a2);
    float rr = logf(r)*2.8f;
    /* ridged multifractal -> filamentary gas */
    float n1 = vnoise3(ca*2.2f, sa*2.2f, rr, seed);
    float n2 = vnoise3(ca*4.7f, sa*4.7f, rr*1.9f+7.f, seed+21);
    float n3 = vnoise3(ca*10.5f, sa*10.5f, rr*3.6f+19.f, seed+43);
    float rid = (1.f-fabsf(2.f*n1-1.f))*0.55f + (1.f-fabsf(2.f*n2-1.f))*0.30f
              + (1.f-fabsf(2.f*n3-1.f))*0.15f;
    float clumps = powf(clampf((rid-0.495f)*2.45f,0.f,1.f), 1.45f);
    float radial = smoothstepf(rin, rin+0.9f, r) * (1.f - smoothstepf(rout-7.0f, rout, r));
    float dens = (0.06f + 0.94f*clumps)*radial;       /* thin haze + filaments */
    if(dens < 0.0025f){ *alpha = 0.f; return v3(0,0,0); }
    float T = clampf((r-rin)/(rout-rin), 0.f, 1.f);
    V3 hot  = v3(1.15f, 1.25f, 1.45f);                /* inner: blue-white  */
    V3 mid  = v3(1.35f, 0.72f, 0.36f);                /* mid: gold          */
    V3 cold = v3(0.62f, 0.16f, 0.09f);                /* outer: deep orange */
    V3 col = T<0.20f ? mixv(hot, mid, T/0.20f) : mixv(mid, cold, (T-0.20f)/0.80f);
    V3 blue = v3(0.30f,0.70f,1.60f);
    col = mixv(col, blue, clampf(tempMix,0.f,1.f)*(1.f-T*0.4f));
    float inner = powf(1.f - T, 4.5f);                /* incandescent inner rim */
    *alpha = clampf(dens*1.15f, 0.f, 1.f);
    return mul(col, dens*(0.26f + inner*1.75f)*bright);
}

/* ------------------------------------------------------------------ */
/* black hole: null geodesics integrated in Cartesian coordinates.
   For a Schwarzschild photon, d2x/dl2 = -3/2 h^2 x / |x|^5 with
   h = |x cross v| conserved - stable for radial and near-radial rays. */
static V3 trace_bh(V3 ro, V3 rd, const float *P, float pixAng){
    float t       = P[P_TIME];
    float diskB   = P[P_DISK];
    float jetB    = P[P_JET];
    float nebA    = P[P_NEB];
    float starB   = P[P_STAR];
    float twk     = P[P_TWK];
    float dens    = P[P_STARDENS];
    float tempMix = P[P_DISKTEMP];
    int   seed    = (int)P[P_SEED];

    V3 pos = ro, dir = rd;
    V3 hv = cross(pos, dir);
    float h2 = dot(hv,hv);

    V3 acc = v3(0,0,0);
    float trans = 1.f;
    int captured = 0;
    float travelled = 0.f;

    const int MAXSTEP = 300;
    for(int i=0;i<MAXSTEP;i++){
        float r2 = dot(pos,pos);
        float r = sqrtf(r2);
        if(r < 1.0f){ captured = 1; break; }
        if(r > 70.f && dot(pos,dir) > 0.f) break;          /* escaped */
        float dt = r < 6.f ? 0.040f*r : (r < 20.f ? 0.085f*r : 0.16f*r);
        if(dt > 6.f) dt = 6.f;
        float inv5 = 1.f/(r2*r2*r);
        V3 a0 = mul(pos, -1.5f*h2*inv5);
        V3 np = add(add(pos, mul(dir,dt)), mul(a0, 0.5f*dt*dt));
        float nr2 = dot(np,np), nr = sqrtf(nr2);
        V3 a1 = mul(np, -1.5f*h2/(nr2*nr2*nr));
        V3 nd = add(dir, mul(add(a0,a1), 0.5f*dt));
        travelled += dt;

        /* --- accretion disk (thin slab at y = 0) --------------------- */
        if(diskB > 0.001f && pos.y*np.y < 0.f){
            float f = pos.y/(pos.y - np.y);
            V3 Xc = add(pos, mul(sub(np,pos), f));
            Xc.y = 0.f;
            float al;
            V3 em = disk_sample(Xc, t, diskB, tempMix, &al, seed);
            if(al > 0.f){
                V3 pd = norm(sub(np,pos));
                /* grazing rays traverse more of the slab */
                float slab = clampf(0.16f/fmaxf(fabsf(pd.y),0.035f), 0.35f, 3.2f);
                al = clampf(al*slab, 0.f, 0.985f);
                float rc = len(Xc);
                V3 rhat = mul(Xc, 1.f/fmaxf(rc,1e-4f));
                V3 vdir = norm(cross(v3(0,1,0), rhat));
                float vmag = sqrtf(0.5f/fmaxf(rc,1.2f));
                float beta = dot(mul(vdir,vmag), mul(pd,-1.f));
                float dop = 1.f/fmaxf(1.f - beta, 0.25f);
                float boost = clampf(powf(dop, 3.1f), 0.10f, 9.f);
                float grav = clampf(1.f - 1.f/fmaxf(rc,1.05f), 0.05f, 1.f);
                boost *= powf(grav, 0.8f);
                /* approaching side also shifts blue, receding side red */
                float sh = clampf((dop-1.f)*1.15f, -0.85f, 0.85f);
                V3 shifted = v3(em.x*(1.f-0.45f*sh), em.y*(1.f-0.05f*sh), em.z*(1.f+0.55f*sh));
                acc = add(acc, mul(shifted, trans*boost*slab));
                trans *= (1.f - al);
                if(trans < 0.02f) break;
            }
        }

        /* --- polar jets: thin, noisy, additive ---------------------- */
        if(jetB > 0.001f){
            float ay = fabsf(pos.y);
            if(ay > 1.6f && ay < 40.f){
                float rho2 = pos.x*pos.x + pos.z*pos.z;
                float cone = 0.17f + ay*0.055f;
                if(rho2 < cone*cone*4.f){
                    float rho = sqrtf(rho2);
                    float prof = expf(-(rho*rho)/(cone*cone));
                    float n = fbm3(v3(pos.x*2.2f, pos.y*0.42f - t*2.4f*(pos.y>0?1.f:-1.f), pos.z*2.2f), 3, seed+404);
                    float e = prof*powf(clampf(n*1.7f-0.42f,0.f,1.f),1.8f)*expf(-ay*0.085f);
                    V3 jc = mixv(v3(0.42f,0.86f,1.55f), v3(0.95f,0.55f,1.45f), clampf(ay*0.032f,0.f,1.f));
                    acc = add(acc, mul(jc, e*jetB*dt*0.55f*trans));
                }
            }
        }

        pos = np; dir = nd;
        if(i == MAXSTEP-1) captured = 1;
    }

    if(captured){
        float ring = powf(clampf((travelled-14.f)*0.05f,0.f,1.f),1.5f)*0.05f;
        return add(acc, mul(v3(0.45f,0.68f,1.3f), ring*P[P_RINGS]));
    }
    V3 ed = norm(dir);
    V3 bg = add(stars(ed, pixAng, t, twk, starB, dens),
                nebula(ed, t, nebA, seed));
    return add(acc, mul(bg, trans));
}

/* ------------------------------------------------------------------ */
/* hyperspace warp: screen-space accelerating light streaks over the
   real star field - continuous lines, no sampling dashes.             */
static V3 warp_streaks(float sx, float sy, float t, float sp){
    float rho = sqrtf(sx*sx+sy*sy);
    if(rho < 1e-4f) rho = 1e-4f;
    float th  = atan2f(sy,sx);
    V3 acc = v3(0,0,0);
    const int   NB[3]   = {440, 210, 96};
    const float BRI[3]  = {0.55f, 0.95f, 1.5f};
    const float WID[3]  = {0.85f, 1.35f, 2.1f};
    for(int L=0;L<3;L++){
        float nb = (float)NB[L];
        float bf = (th/6.2831853f + 0.5f)*nb;
        int ib = (int)floorf(bf);
        for(int k=-1;k<=1;k++){
            int bin = ib+k; int wrapped = ((bin % NB[L]) + NB[L]) % NB[L];
            float off = h2i(wrapped,0,L*7717+3);
            if(off > 0.78f) continue;                       /* empty slot */
            float ph  = h2i(wrapped,1,L*7717+5);
            float spd = 0.55f + 0.9f*h2i(wrapped,2,L*7717+9);
            float mag = h2i(wrapped,3,L*7717+11);
            float slot = (float)(bin) + h2i(wrapped,4,L*7717+13);
            float dth = (bf - slot)*6.2831853f/nb;          /* angular offset  */
            float perp = fabsf(dth)*rho;                    /* arc distance    */
            float w = 0.0034f*WID[L]*(0.55f+rho*0.55f);
            float g = expf(-(perp*perp)/(w*w));
            if(g < 0.004f) continue;
            float u = ph + t*spd*sp*0.30f;
            u -= floorf(u);
            float head = 0.035f*expf(u*4.6f);               /* accelerating    */
            float tail = head*(0.12f + 0.75f*sp);
            if(rho > head || rho < head-tail) continue;
            float along = (head-rho)/fmaxf(tail,1e-4f);     /* 0 head .. 1 tail*/
            float fade = powf(1.f-along, 0.6f)*(0.25f+0.75f*smoothstepf(0.f,0.28f,rho));
            float b = powf(mag,2.4f)*BRI[L]*fade*g;
            V3 tint = mixv(v3(0.85f,0.93f,1.30f), v3(1.30f,0.70f,1.05f), along*0.8f);
            acc = add(acc, mul(tint, b*2.3f));
        }
    }
    return acc;
}
static V3 trace_warp(V3 rd, float sx, float sy, const float *P, float pixAng){
    float t = P[P_TIME], sp = P[P_WARP];
    V3 acc = mul(stars(rd, pixAng, t, P[P_TWK], P[P_STAR]*0.65f, P[P_STARDENS]), 1.f);
    acc = add(acc, mul(nebula(rd, t, P[P_NEB]*0.85f, (int)P[P_SEED]), 0.9f));
    acc = add(acc, warp_streaks(sx, sy, t, sp));
    float rho = sqrtf(sx*sx+sy*sy);
    acc = add(acc, mul(v3(0.30f,0.52f,1.10f), expf(-rho*7.0f)*0.42f*sp));
    acc = add(acc, mul(v3(1.0f,0.92f,1.0f),  expf(-rho*26.f)*1.25f*sp));
    return acc;
}

/* interior of the singularity: collapsing filament tunnel            */
static V3 trace_tunnel(float sx, float sy, const float *P){
    float t = P[P_TIME], k = P[P_TUNNEL];
    float r = sqrtf(sx*sx+sy*sy) + 1e-4f;
    float ang = atan2f(sy,sx);
    float lr = logf(r);
    float z = lr*2.4f - t*1.9f*k;
    float a = ang + lr*1.9f;
    float ca = cosf(a), sa = sinf(a);
    int sd = (int)P[P_SEED];
    float m1 = vnoise3(ca*2.6f, sa*2.6f, z, sd+13);
    float m2 = vnoise3(ca*5.9f, sa*5.9f, z*2.1f+5.f, sd+91);
    float m3 = vnoise3(ca*12.3f, sa*12.3f, z*4.3f+17.f, sd+131);
    float m4 = vnoise3(ca*24.0f, sa*24.0f, z*7.5f+41.f, sd+177);
    float rid = (1.f-fabsf(2*m1-1.f))*0.46f + (1.f-fabsf(2*m2-1.f))*0.28f
              + (1.f-fabsf(2*m3-1.f))*0.16f + (1.f-fabsf(2*m4-1.f))*0.10f;
    float fil = powf(clampf((rid-0.63f)*4.2f,0.f,1.f),1.7f);
    /* large scale voids so the tunnel has black between the filaments */
    float voids = powf(clampf((vnoise3(ca*1.15f, sa*1.15f, z*0.55f, sd+61)-0.30f)*2.2f,0.f,1.f), 0.9f);
    fil *= 0.25f + 1.15f*voids;
    float depth = powf(clampf(1.f-r*0.60f,0.f,1.f), 1.25f);
    float band = clampf(0.5f+0.5f*sinf(z*0.42f+t*0.4f),0.f,1.f);
    V3 col = mixv(v3(0.16f,0.48f,1.40f), v3(1.15f,0.26f,0.78f), powf(band, 3.4f));
    col = mixv(col, v3(0.85f,0.96f,1.20f), powf(clampf(depth*1.15f,0.f,1.f), 2.0f)*0.55f);
    V3 acc = mul(col, fil*(0.30f+2.10f*depth));
    /* light rays converging on the core */
    float rays = powf(clampf(vnoise3(cosf(ang)*7.5f, sinf(ang)*7.5f, t*0.55f, sd+7)*1.9f-0.75f,0.f,1.f),2.6f);
    acc = add(acc, mul(v3(0.55f,0.78f,1.35f), rays*depth*depth*0.85f));
    acc = add(acc, mul(v3(1.30f,1.15f,1.55f), powf(clampf(1.f-r*3.6f,0.f,1.f), 3.4f)*1.35f));
    acc = add(acc, mul(v3(0.30f,0.55f,1.25f), expf(-r*6.5f)*0.28f));
    return acc;
}

/* ------------------------------------------------------------------ */
/* other subjects: a ringed gas giant, a star, a nebula fly-through.
   The edit needs to cut between different things, not only between
   angles on the same thing.                                          */

static int sphere_hit(V3 ro, V3 rd, float R, float *tout){
    float b = dot(ro,rd), c = dot(ro,ro) - R*R;
    float h = b*b - c;
    if(h < 0.f) return 0;
    h = sqrtf(h);
    float t = -b - h;
    if(t < 0.f) t = -b + h;
    if(t < 0.f) return 0;
    *tout = t; return 1;
}

/* density of the ring system at cylindrical radius r */
static float ring_density(float r, float rin, float rout, int seed){
    if(r < rin || r > rout) return 0.f;
    float u = (r - rin)/(rout - rin);
    float n  = vnoise2(u*26.f, 0.5f, seed);
    float n2 = vnoise2(u*74.f, 3.5f, seed+9);
    float d = 0.45f + 0.75f*n - 0.35f*n2;
    /* Cassini-style gaps */
    d *= 1.f - 0.92f*expf(-powf((u-0.42f)/0.035f, 2.f));
    d *= 1.f - 0.75f*expf(-powf((u-0.70f)/0.028f, 2.f));
    d *= smoothstepf(0.f,0.06f,u)*(1.f - smoothstepf(0.88f,1.f,u));
    return clampf(d, 0.f, 1.f);
}

static V3 planet_surface(V3 n, float t, int seed, float *rough){
    /* bands: noise stretched hard in latitude, drifting in longitude */
    V3 q = v3(n.x*0.75f, n.y*7.5f, n.z*0.75f);
    float drift = t*0.035f;
    float w = fbm3(v3(n.x*1.7f + drift, n.y*2.2f, n.z*1.7f), 3, seed+5);
    float band = fbm3(add(q, v3(w*1.6f, 0.f, w*1.6f)), 4, seed);
    float fine = fbm3(v3(n.x*3.2f + drift*2.f, n.y*22.f, n.z*3.2f), 3, seed+31);
    float v = clampf(band*0.78f + fine*0.32f, 0.f, 1.f);
    /* one storm */
    V3 sc = norm(v3(0.55f, -0.34f, 0.76f));
    float sd = 1.f - clampf(dot(n, sc), 0.f, 1.f);
    float storm = expf(-sd*44.f);
    int pal = seed % 3;
    V3 a, b, c;
    if(pal == 0){ a=v3(0.26f,0.11f,0.06f); b=v3(0.74f,0.46f,0.24f); c=v3(0.92f,0.82f,0.68f); }
    else if(pal == 1){ a=v3(0.05f,0.11f,0.22f); b=v3(0.22f,0.44f,0.64f); c=v3(0.74f,0.85f,0.95f); }
    else { a=v3(0.18f,0.08f,0.20f); b=v3(0.52f,0.28f,0.54f); c=v3(0.84f,0.74f,0.84f); }
    V3 col = v < 0.5f ? mixv(a, b, v*2.f) : mixv(b, c, (v-0.5f)*2.f);
    col = mixv(col, v3(0.95f,0.72f,0.55f), storm*0.85f);
    *rough = v;
    return col;
}

static V3 trace_planet(V3 ro, V3 rd, const float *P, float pixAng){
    float t = P[P_TIME];
    int seed = (int)P[P_SEED] + 77;
    const float R = 1.0f;
    int has_rings = ((seed/3) % 5) != 0;
    float rin = 1.34f + 0.16f*h1((unsigned)seed*2654435761u);
    float rout = rin + 0.85f + 0.55f*h1((unsigned)(seed*40503u+7u));
    if(!has_rings){ rin = 99.f; rout = 99.f; }
    float laz = P[P_LIGHTAZ]*(float)M_PI/180.f, lel = P[P_LIGHTEL]*(float)M_PI/180.f;
    V3 L = norm(v3(cosf(lel)*sinf(laz), sinf(lel), -cosf(lel)*cosf(laz)));
    float ts, tr;
    int hs = sphere_hit(ro, rd, R, &ts);
    int hr = 0;
    /* rings live in the planet's equatorial plane, y = 0 */
    if(fabsf(rd.y) > 1e-5f){
        tr = -ro.y/rd.y;
        if(tr > 0.f){
            V3 pr = add(ro, mul(rd, tr));
            float rr = sqrtf(pr.x*pr.x + pr.z*pr.z);
            if(rr > rin && rr < rout) hr = 1;
        }
    }
    V3 bg = add(stars(rd, pixAng, t, P[P_TWK], P[P_STAR], P[P_STARDENS]),
                nebula(rd, t, P[P_NEB], (int)P[P_SEED]));
    V3 col = bg;

    /* --- planet body ------------------------------------------------- */
    if(hs){
        V3 p = add(ro, mul(rd, ts));
        V3 n = norm(p);
        float rough;
        V3 sc = planet_surface(n, t, seed, &rough);
        float ndl = dot(n, L);
        float lit = smoothstepf(-0.06f, 0.42f, ndl);
        lit *= 0.35f + 0.65f*powf(clampf(ndl,0.f,1.f), 0.75f);   /* falloff zum Terminator */
        /* the rings drop a shadow on the planet */
        if(fabsf(L.y) > 1e-4f){
            float th = -p.y/L.y;
            if(th > 0.f){
                V3 ph = add(p, mul(L, th));
                float rr = sqrtf(ph.x*ph.x + ph.z*ph.z);
                lit *= 1.f - 0.78f*ring_density(rr, rin, rout, seed);
            }
        }
        float amb = 0.028f;
        V3 body = mul(sc, lit*1.15f + amb);
        /* limb haze */
        float fres = powf(1.f - clampf(dot(n, mul(rd,-1.f)), 0.f, 1.f), 3.2f);
        V3 atmo = mixv(v3(0.35f,0.55f,1.05f), v3(1.05f,0.72f,0.45f), 0.35f);
        body = add(body, mul(atmo, fres*(0.20f + 0.80f*lit)*1.05f));
        col = body;
    } else {
        /* atmosphere glow for rays that graze the limb */
        float b = -dot(ro, rd);
        V3 cp = add(ro, mul(rd, fmaxf(b, 0.f)));
        float d = len(cp);
        if(d > R && b > 0.f){
            float g = expf(-(d - R)*13.f);
            float side = clampf(dot(norm(cp), L)*0.5f + 0.5f, 0.f, 1.f);
            col = add(col, mul(v3(0.42f,0.62f,1.10f), g*0.55f*(0.25f + 0.75f*side)));
        }
    }

    /* --- rings, composited in front of or behind the planet ---------- */
    if(hr && (!hs || tr < ts)){
        V3 pr = add(ro, mul(rd, tr));
        float rr = sqrtf(pr.x*pr.x + pr.z*pr.z);
        float dens = ring_density(rr, rin, rout, seed);
        if(dens > 0.002f){
            float u = (rr - rin)/(rout - rin);
            V3 rc = mixv(v3(0.78f,0.68f,0.55f), v3(0.95f,0.90f,0.86f), u);
            rc = mixv(rc, v3(0.60f,0.52f,0.62f), 0.35f*vnoise2(u*40.f, 1.5f, seed+3));
            /* planet shadow on the rings */
            float shadow = 1.f;
            float tb;
            if(sphere_hit(pr, L, R, &tb)) shadow = 0.13f;
            /* grazing view thickens the ring */
            float thick = clampf(0.30f/fmaxf(fabsf(rd.y), 0.05f), 0.4f, 2.6f);
            float a = clampf(dens*thick*0.95f, 0.f, 1.f);
            float rlit = 0.25f + 0.75f*clampf(fabsf(L.y)*2.2f, 0.f, 1.f);
            V3 em = mul(rc, shadow*rlit*(1.05f + 0.35f*thick));
            col = add(mul(col, 1.f - a), mul(em, a));
        }
    }
    return col;
}

static V3 core_rim(int seed){
    switch(seed % 4){
        case 0: return v3(1.5f,0.85f,0.45f);
        case 1: return v3(0.75f,1.05f,1.70f);
        case 2: return v3(1.5f,0.55f,0.25f);
        default:return v3(1.4f,0.95f,1.10f);
    }
}
static V3 trace_star(V3 ro, V3 rd, const float *P, float pixAng){
    float t = P[P_TIME];
    int seed = (int)P[P_SEED] + 211;
    const float R = 1.0f;
    float ts;
    V3 col = mul(stars(rd, pixAng, t, P[P_TWK], P[P_STAR]*0.5f, P[P_STARDENS]), 1.f);
    col = add(col, mul(nebula(rd, t, P[P_NEB]*0.5f, (int)P[P_SEED]), 0.6f));
    float b = -dot(ro, rd);
    V3 cp = add(ro, mul(rd, fmaxf(b, 0.f)));
    float d = fmaxf(len(cp), R*1.001f);
    if(sphere_hit(ro, rd, R, &ts)){
        V3 p = add(ro, mul(rd, ts));
        V3 n = norm(p);
        float gran = fbm3(add(mul(n, 16.f), v3(0.f, t*0.22f, 0.f)), 4, seed);
        float fine = fbm3(add(mul(n, 46.f), v3(t*0.14f, 0.f, 0.f)), 3, seed+7);
        float v = clampf((gran*0.68f + fine*0.42f - 0.34f)*1.55f, 0.f, 1.f);
        float mu = clampf(dot(n, mul(rd,-1.f)), 0.f, 1.f);
        float limb = 0.32f + 0.68f*powf(mu, 0.55f);        /* limb darkening */
        V3 edge, core;
        switch(seed % 4){                                  /* spectral class */
            case 0: edge=v3(1.40f,0.78f,0.26f); core=v3(1.75f,1.58f,1.10f); break; /* G, gold */
            case 1: edge=v3(0.55f,0.85f,1.55f); core=v3(1.25f,1.55f,1.85f); break; /* B, blue */
            case 2: edge=v3(1.35f,0.34f,0.14f); core=v3(1.60f,0.85f,0.42f); break; /* K, orange */
            default:edge=v3(1.20f,0.62f,0.72f); core=v3(1.70f,1.45f,1.55f); break; /* pink-white */
        }
        V3 s = mixv(edge, core, powf(v, 1.15f));
        float spot = smoothstepf(0.06f, 0.005f, v);         /* rare, small */
        s = mul(s, 1.f - 0.30f*spot);
        col = mul(s, limb*(0.72f + 0.34f*v)*P[P_DISK]*0.78f);
        /* chromosphere at the very edge */
        col = add(col, mul(v3(1.3f,0.35f,0.22f), powf(1.f-mu, 5.f)*0.5f*P[P_DISK]));
    } else if(b > 0.f){
        float x = d/R;
        float corona = powf(1.f/x, 2.3f);
        V3 nd = norm(cp);
        float str = fbm3(add(mul(nd, 6.5f), v3(0.f, 0.f, t*0.12f)), 3, seed+19);
        float ray = powf(clampf(str*1.7f-0.42f,0.f,1.f), 1.3f);
        corona *= 0.30f + 1.55f*ray;
        V3 ca_, cb_;
        switch(seed % 4){
            case 0: ca_=v3(1.45f,0.72f,0.30f); cb_=v3(0.95f,0.42f,0.75f); break;
            case 1: ca_=v3(0.60f,0.95f,1.60f); cb_=v3(0.55f,0.55f,1.15f); break;
            case 2: ca_=v3(1.40f,0.45f,0.18f); cb_=v3(0.95f,0.30f,0.42f); break;
            default:ca_=v3(1.30f,0.70f,0.85f); cb_=v3(0.80f,0.55f,1.05f); break;
        }
        V3 cc = mixv(ca_, cb_, clampf((x-1.f)*0.45f,0.f,1.f));
        col = add(col, mul(cc, corona*0.95f*P[P_DISK]));
        /* thin bright rim right at the limb */
        col = add(col, mul(core_rim(seed), expf(-(x-1.f)*46.f)*0.9f*P[P_DISK]));
    }
    return col;
}

static V3 trace_nebula_fly(V3 ro, V3 rd, const float *P, float pixAng){
    float t = P[P_TIME];
    int seed = (int)P[P_SEED] + 401;
    V3 acc = v3(0,0,0);
    float trans = 1.f;
    const int STEPS = 44;
    const float dt = 0.55f;
    float jitter = h1((unsigned)((int)(rd.x*9871.f) ^ (int)(rd.y*1237.f)))*dt;
    for(int i=0;i<STEPS;i++){
        float s = 0.6f + i*dt + jitter;
        V3 p = add(ro, mul(rd, s));
        float d = fbm3(mul(p, 0.30f), 4, seed);
        float m = fbm3(mul(p, 0.10f), 3, seed+61);
        float fil = 1.f - fabsf(2.f*vnoise3(p.x*0.85f, p.y*0.85f, p.z*0.85f, seed+13) - 1.f);
        float dens = powf(clampf((d-0.545f)*3.4f,0.f,1.f), 2.0f)
                   * powf(clampf((m-0.44f)*2.8f,0.f,1.f), 1.3f)
                   * (0.25f + 1.25f*powf(clampf((fil-0.45f)*2.2f,0.f,1.f), 1.5f));
        if(dens > 0.002f){
            float hot = powf(clampf((d-0.60f)*3.2f,0.f,1.f), 2.f);
            V3 c1 = v3(0.08f,0.24f,0.72f), c2 = v3(0.14f,0.52f,0.62f), c3 = v3(1.15f,0.55f,0.62f);
            float tint = fbm3(mul(p, 0.055f), 2, seed+91);
            V3 c = mixv(c1, c2, clampf(tint*2.2f-0.55f,0.f,1.f));
            c = mixv(c, c3, hot*0.85f);
            float e = dens*dt*0.85f;
            acc = add(acc, mul(c, e*trans*1.45f));
            trans *= expf(-dens*dt*2.6f);
            if(trans < 0.02f) break;
        }
    }
    V3 bg = add(stars(rd, pixAng, t, P[P_TWK], P[P_STAR], P[P_STARDENS]),
                nebula(rd, t, P[P_NEB]*0.45f, (int)P[P_SEED]));
    return add(acc, mul(bg, trans));
}

/* ------------------------------------------------------------------ */
/* colour pipeline                                                     */
static V3 aces(V3 x){
    V3 r;
    r.x = clampf((x.x*(2.51f*x.x+0.03f))/(x.x*(2.43f*x.x+0.59f)+0.14f),0.f,1.f);
    r.y = clampf((x.y*(2.51f*x.y+0.03f))/(x.y*(2.43f*x.y+0.59f)+0.14f),0.f,1.f);
    r.z = clampf((x.z*(2.51f*x.z+0.03f))/(x.z*(2.43f*x.z+0.59f)+0.14f),0.f,1.f);
    return r;
}
static inline float srgb(float c){
    return c <= 0.0031308f ? 12.92f*c : 1.055f*powf(c,1.f/2.4f)-0.055f;
}

/* bilinear fetch from the float RGB buffer */
static inline V3 fetch(const float *buf, int W, int H, float x, float y){
    x = clampf(x, 0.f, W-1.001f); y = clampf(y, 0.f, H-1.001f);
    int ix=(int)x, iy=(int)y; float fx=x-ix, fy=y-iy;
    int i00=(iy*W+ix)*3, i10=i00+3, i01=i00+W*3, i11=i01+3;
    V3 r;
    r.x = mixf(mixf(buf[i00+0],buf[i10+0],fx), mixf(buf[i01+0],buf[i11+0],fx), fy);
    r.y = mixf(mixf(buf[i00+1],buf[i10+1],fx), mixf(buf[i01+1],buf[i11+1],fx), fy);
    r.z = mixf(mixf(buf[i00+2],buf[i10+2],fx), mixf(buf[i01+2],buf[i11+2],fx), fy);
    return r;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv){
    if(argc < 4){ fprintf(stderr,"usage: render timeline.csv W H [first last] [texlist]\n"); return 1; }
    const char *tlpath = argv[1];
    int W = atoi(argv[2]), H = atoi(argv[3]);
    int first = argc>4 ? atoi(argv[4]) : 0;
    int last  = argc>5 ? atoi(argv[5]) : -1;
    const char *texlist = argc>6 ? argv[6] : NULL;

    /* ---- load timeline ---- */
    FILE *f = fopen(tlpath,"r");
    if(!f){ fprintf(stderr,"cannot open %s\n", tlpath); return 1; }
    int cap = 1024, n = 0;
    Frame *fr = malloc(cap*sizeof(Frame));
    char line[4096];
    while(fgets(line,sizeof line,f)){
        if(line[0]=='#') continue;
        if(n>=cap){ cap*=2; fr = realloc(fr, cap*sizeof(Frame)); }
        char *p = line; int k=0;
        while(k<NPARAM && *p){
            fr[n].p[k++] = strtof(p,&p);
            if(*p==',') p++;
        }
        for(; k<NPARAM; k++) fr[n].p[k]=0.f;
        n++;
    }
    fclose(f);
    if(last < 0 || last >= n) last = n-1;
    fprintf(stderr,"timeline: %d frames, rendering %d..%d at %dx%d\n", n, first, last, W, H);

    /* ---- load text textures (raw: int32 w, int32 h, RGBA) ---- */
    if(texlist){
        FILE *tl = fopen(texlist,"r");
        if(tl){
            char path[1024];
            while(fgets(path,sizeof path,tl) && g_ntex<MAXTEX){
                path[strcspn(path,"\r\n")] = 0;
                if(!path[0]) continue;
                FILE *tf = fopen(path,"rb");
                if(!tf){ fprintf(stderr,"tex missing: %s\n",path); continue; }
                int wh[2]; if(fread(wh,4,2,tf)!=2){ fclose(tf); continue; }
                size_t sz = (size_t)wh[0]*wh[1]*4;
                unsigned char *px = malloc(sz);
                if(fread(px,1,sz,tf)!=sz){ free(px); fclose(tf); continue; }
                fclose(tf);
                g_tex[g_ntex].w=wh[0]; g_tex[g_ntex].h=wh[1]; g_tex[g_ntex].px=px;
                g_ntex++;
            }
            fclose(tl);
        }
    }

    size_t npx = (size_t)W*H;
    float *buf  = malloc(npx*3*sizeof(float));
    float *post = malloc(npx*3*sizeof(float));
    int BW = W/4, BH = H/4;
    float *bl0 = malloc((size_t)BW*BH*3*sizeof(float));
    float *bl1 = malloc((size_t)BW*BH*3*sizeof(float));
    unsigned char *out = malloc(npx*3);

    for(int fi=first; fi<=last; fi++){
        const float *P = fr[fi].p;
        int mode = (int)(P[P_MODE]+0.5f);
        float fov = P[P_FOV]*(float)M_PI/180.f;
        float tanH = tanf(fov*0.5f);
        float aspect = (float)W/(float)H;
        float pixAng = fov/(float)H;

        V3 ro = v3(P[P_CX],P[P_CY],P[P_CZ]);
        V3 ta = v3(P[P_TX],P[P_TY],P[P_TZ]);
        V3 fw = norm(sub(ta,ro));
        V3 up0 = v3(0,1,0);
        if(fabsf(dot(fw,up0)) > 0.995f) up0 = v3(0,0,1);
        V3 rt = norm(cross(fw,up0));
        V3 up = cross(rt,fw);
        float rl = P[P_ROLL];
        V3 rt2 = add(mul(rt,cosf(rl)), mul(up,sinf(rl)));
        V3 up2 = sub(mul(up,cosf(rl)), mul(rt,sinf(rl)));

        float shx = P[P_SHX], shy = P[P_SHY];

        #pragma omp parallel for schedule(dynamic,8)
        for(int y=0;y<H;y++){
            for(int x=0;x<W;x++){
                float sx = ((x+0.5f)/W*2.f-1.f)*aspect + shx;
                float sy = (1.f-(y+0.5f)/H*2.f) + shy;
                V3 col;
                if(mode == 2){
                    float rsx = sx*cosf(rl) - sy*sinf(rl);
                    float rsy = sx*sinf(rl) + sy*cosf(rl);
                    col = trace_tunnel(rsx*1.05f, rsy*1.05f, P);
                } else {
                    V3 rd = norm(add(add(mul(rt2, sx*tanH), mul(up2, sy*tanH)), fw));
                    switch(mode){
                        case 1:  col = trace_warp(rd, sx, sy, P, pixAng); break;
                        case 3:  col = trace_planet(ro, rd, P, pixAng); break;
                        case 4:  col = trace_star(ro, rd, P, pixAng); break;
                        case 5:  col = trace_nebula_fly(ro, rd, P, pixAng); break;
                        default: col = trace_bh(ro, rd, P, pixAng); break;
                    }
                }
                size_t o = ((size_t)y*W+x)*3;
                buf[o+0]=col.x; buf[o+1]=col.y; buf[o+2]=col.z;
            }
        }

        /* ---- bloom: threshold, downsample 4x, separable blur ---- */
        float bloomAmt = P[P_BLOOM];
        if(bloomAmt > 0.001f){
            #pragma omp parallel for schedule(static)
            for(int y=0;y<BH;y++) for(int x=0;x<BW;x++){
                float r=0,g=0,b=0;
                for(int j=0;j<4;j++) for(int i=0;i<4;i++){
                    size_t o=((size_t)(y*4+j)*W + (x*4+i))*3;
                    r+=buf[o]; g+=buf[o+1]; b+=buf[o+2];
                }
                r/=16; g/=16; b/=16;
                float l = 0.2126f*r+0.7152f*g+0.0722f*b;
                float w = clampf((l-0.85f)*0.75f,0.f,5.f);
                size_t q=((size_t)y*BW+x)*3;
                bl0[q]=r*w; bl0[q+1]=g*w; bl0[q+2]=b*w;
            }
            const int R = 10;
            static float kern[21]; static int kinit=0;
            if(!kinit){ float s=0; for(int i=-R;i<=R;i++){ kern[i+R]=expf(-0.5f*(i*i)/(R*0.45f*R*0.45f)); s+=kern[i+R]; }
                        for(int i=0;i<2*R+1;i++) kern[i]/=s; kinit=1; }
            #pragma omp parallel for schedule(static)
            for(int y=0;y<BH;y++) for(int x=0;x<BW;x++){
                float r=0,g=0,b=0;
                for(int i=-R;i<=R;i++){
                    int xx = x+i; if(xx<0)xx=0; if(xx>=BW)xx=BW-1;
                    size_t q=((size_t)y*BW+xx)*3; float k=kern[i+R];
                    r+=bl0[q]*k; g+=bl0[q+1]*k; b+=bl0[q+2]*k;
                }
                size_t q=((size_t)y*BW+x)*3; bl1[q]=r; bl1[q+1]=g; bl1[q+2]=b;
            }
            #pragma omp parallel for schedule(static)
            for(int y=0;y<BH;y++) for(int x=0;x<BW;x++){
                float r=0,g=0,b=0;
                for(int i=-R;i<=R;i++){
                    int yy=y+i; if(yy<0)yy=0; if(yy>=BH)yy=BH-1;
                    size_t q=((size_t)yy*BW+x)*3; float k=kern[i+R];
                    r+=bl1[q]*k; g+=bl1[q+1]*k; b+=bl1[q+2]*k;
                }
                size_t q=((size_t)y*BW+x)*3; bl0[q]=r; bl0[q+1]=g; bl0[q+2]=b;
            }
            #pragma omp parallel for schedule(static)
            for(int y=0;y<H;y++){
                float fy = (float)y/4.f;
                for(int x=0;x<W;x++){
                    float fx=(float)x/4.f;
                    V3 bv = fetch(bl0,BW,BH,fx,fy);
                    size_t o=((size_t)y*W+x)*3;
                    buf[o+0]+=bv.x*bloomAmt; buf[o+1]+=bv.y*bloomAmt; buf[o+2]+=bv.z*bloomAmt;
                }
            }
        }

        /* ---- radial zoom blur + chromatic aberration + zoom ---- */
        float zoom = P[P_ZOOM] > 0.01f ? P[P_ZOOM] : 1.f;
        float rb = P[P_RBLUR], ca = P[P_CHROMA];
        float spin = P[P_SPIN];                 /* rotational smear - whip cuts */
        float glitch = P[P_GLITCH];             /* horizontal slice tear        */
        float mirror = P[P_MIRROR];             /* 1 = mirror x, 2 = quad       */
        int taps = (rb > 0.002f || fabsf(spin) > 0.002f) ? 9 : 1;
        float cxp = W*0.5f, cyp = H*0.5f;
        unsigned int gseed = (unsigned)(fi*2246822519u);
        #pragma omp parallel for schedule(static)
        for(int y=0;y<H;y++){
            /* slice tear: whole bands of rows jump sideways for a frame */
            float tear = 0.f, tearc = 0.f;
            if(glitch > 0.001f){
                int band = (int)(y/(float)H*24.f);
                float r1 = h2i(band, 0, (int)(gseed & 0xffff));
                if(r1 < 0.55f){
                    tear = (h2i(band,1,(int)(gseed&0xffff))-0.5f)*glitch*W*0.35f;
                    tearc = (h2i(band,2,(int)(gseed&0xffff))-0.5f)*glitch*22.f;
                }
            }
            for(int x=0;x<W;x++){
                float dx = (x-cxp), dy = (y-cyp);
                if(mirror > 0.5f){
                    dx = fabsf(dx);
                    if(mirror > 1.5f) dy = fabsf(dy);
                }
                float acc[3]={0,0,0};
                for(int ch=0; ch<3; ch++){
                    float cscale = 1.f + ca*((ch==0)?1.f:(ch==2)?-1.f:0.f);
                    float chshift = tear + tearc*((ch==0)?1.f:(ch==2)?-1.f:0.f);
                    float sum=0, wsum=0;
                    for(int tp=0; tp<taps; tp++){
                        float f = taps>1 ? (float)tp/(taps-1) : 0.f;
                        float sc = (1.f/zoom) * cscale * (1.f - rb*0.055f*f);
                        float a = spin*f;
                        float ca_ = cosf(a), sa_ = sinf(a);
                        float rx = dx*ca_ - dy*sa_, ry = dx*sa_ + dy*ca_;
                        float sxp = cxp + rx*sc + chshift, syp = cyp + ry*sc;
                        float w = 1.f - 0.55f*f;
                        V3 v = fetch(buf,W,H,sxp,syp);
                        sum += ((ch==0)?v.x:(ch==1)?v.y:v.z)*w; wsum += w;
                    }
                    acc[ch] = sum/wsum;
                }
                size_t o=((size_t)y*W+x)*3;
                post[o]=acc[0]; post[o+1]=acc[1]; post[o+2]=acc[2];
            }
        }

        /* ---- grade, vignette, grain, text, fade ---- */
        float expo = P[P_EXPO], flash = P[P_FLASH], vign = P[P_VIGN];
        float grain = P[P_GRAIN], sat = P[P_SAT], contrast = P[P_CONTRAST];
        float fade = P[P_FADE], inv = P[P_INVERT];
        int tix = (int)P[P_TEXT];
        float talpha = P[P_TALPHA], tscale = (P[P_TSCALE] > 0.01f ? P[P_TSCALE] : 1.f)*((float)W/1080.f);
        float tyoff = P[P_TOFF];
        unsigned int fseed = (unsigned)fi*2654435761u;

        #pragma omp parallel for schedule(static)
        for(int y=0;y<H;y++){
            for(int x=0;x<W;x++){
                size_t o=((size_t)y*W+x)*3;
                V3 c = v3(post[o],post[o+1],post[o+2]);
                c = mul(c, expo*(1.f + 0.85f*flash));
                c = aces(c);
                if(flash > 0.22f){
                    float fw = clampf((flash-0.22f)*1.45f, 0.f, 1.f);
                    fw *= fw;
                    c = mixv(c, v3(1.f,0.995f,1.f), fw);
                }
                /* colour grade: cool shadows, warm-magenta highlights */
                float l = 0.2126f*c.x+0.7152f*c.y+0.0722f*c.z;
                V3 shadow = v3(-0.012f, 0.004f, 0.030f);
                V3 high   = v3( 0.030f,-0.004f, 0.020f);
                c = add(c, add(mul(shadow,(1.f-l)*(1.f-l)), mul(high, l*l)));
                c = mixv(v3(l,l,l), c, sat);
                c = add(v3(0.42f,0.42f,0.42f), mul(sub(c,v3(0.42f,0.42f,0.42f)), contrast));
                if(inv > 0.001f){
                    V3 ic = v3(1.f-c.x,1.f-c.y,1.f-c.z);
                    c = mixv(c, ic, inv);
                }
                /* vignette */
                float nx=(x/(float)W-0.5f)*2.f, ny=(y/(float)H-0.5f)*2.f;
                float rr = nx*nx*0.85f + ny*ny*0.55f;
                c = mul(c, 1.f - vign*clampf(rr*0.62f,0.f,1.f));
                /* text overlay */
                if(tix >= 0 && tix < g_ntex && talpha > 0.002f){
                    Tex *T = &g_tex[tix];
                    float tw = T->w*tscale, th = T->h*tscale;
                    float ox = (W-tw)*0.5f, oy = (H-th)*0.5f + tyoff*H;
                    float u = (x-ox)/fmaxf(tscale,1e-4f), v = (y-oy)/fmaxf(tscale,1e-4f);
                    if(u>=0 && v>=0 && u<T->w-1 && v<T->h-1){
                        int iu=(int)u, iv=(int)v; float fu=u-iu, fv=v-iv;
                        float acc[4]={0,0,0,0};
                        for(int ch=0;ch<4;ch++){
                            float p00=T->px[((size_t)iv*T->w+iu)*4+ch];
                            float p10=T->px[((size_t)iv*T->w+iu+1)*4+ch];
                            float p01=T->px[((size_t)(iv+1)*T->w+iu)*4+ch];
                            float p11=T->px[((size_t)(iv+1)*T->w+iu+1)*4+ch];
                            acc[ch]=mixf(mixf(p00,p10,fu),mixf(p01,p11,fu),fv)/255.f;
                        }
                        float al = acc[3]*talpha;
                        c = mixv(c, v3(acc[0],acc[1],acc[2]), al);
                    }
                }
                /* grain */
                if(grain > 0.0005f){
                    /* grain only where it does something - midtones and gradients.
                       Grain on near-black pixels is invisible but very expensive
                       to encode, so weight it by l*(1-l). */
                    float g = (h1(uhash((unsigned)((x>>1)*1973+(y>>1)*9277)) ^ fseed)-0.5f);
                    float amt = grain*(0.10f + 3.0f*l*(1.f-l));
                    c = add(c, v3(g*amt, g*amt*0.94f, g*amt*1.06f));
                }
                c = mul(c, 1.f-fade);
                size_t q=o;
                out[q+0]=(unsigned char)(clampf(srgb(clampf(c.x,0.f,1.f)),0.f,1.f)*255.f+0.5f);
                out[q+1]=(unsigned char)(clampf(srgb(clampf(c.y,0.f,1.f)),0.f,1.f)*255.f+0.5f);
                out[q+2]=(unsigned char)(clampf(srgb(clampf(c.z,0.f,1.f)),0.f,1.f)*255.f+0.5f);
            }
        }
        fwrite(out,1,npx*3,stdout);
        if(((fi-first) % 30)==0) fprintf(stderr,"\rframe %d/%d", fi, last);
    }
    fprintf(stderr,"\ndone\n");
    return 0;
}
