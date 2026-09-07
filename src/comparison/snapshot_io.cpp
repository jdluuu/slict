#include "slict/solver_comparison.h"
#include <Eigen/Geometry>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
#include <unistd.h>

namespace slict::comparison {
namespace {
std::uint64_t Hash(const std::string& bytes) {
  std::uint64_t hash=14695981039346656037ULL;
  for (unsigned char c:bytes) {hash^=c;hash*=1099511628211ULL;}return hash;
}
std::string Hex(std::uint64_t value) {std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<value;return out.str();}
struct Writer {
  std::string bytes;
  void Integer(std::uint64_t value) {for (int i=0;i<8;++i) bytes.push_back(static_cast<char>((value>>(8*i))&255));}
  void Double(double value) {static_assert(sizeof(double)==8);std::uint64_t bits;std::memcpy(&bits,&value,8);Integer(bits);}
  void Vector(const Vector3& v) {for (int i=0;i<3;++i) Double(v[i]);}
  void Controls(const State& state) {
    Integer(state.rotations.size());
    for (std::size_t i=0;i<state.rotations.size();++i) {
      const auto q=state.rotations[i].unit_quaternion();for (int j=0;j<4;++j) Double(q.coeffs()[j]);Vector(state.positions[i]);
    }
    Vector(state.gyro_bias);Vector(state.accel_bias);
  }
  void Matrix(const Eigen::MatrixXd& matrix) {
    Integer(matrix.rows());Integer(matrix.cols());
    for (int r=0;r<matrix.rows();++r) for (int c=0;c<matrix.cols();++c) Double(matrix(r,c));
  }
};
struct Reader {
  const std::string& bytes;std::size_t position=0;
  std::uint64_t Integer() {
    if (bytes.size()-position<8) throw std::runtime_error("Truncated snapshot");
    std::uint64_t value=0;for (int i=0;i<8;++i) value|=std::uint64_t(static_cast<unsigned char>(bytes[position++]))<<(8*i);return value;
  }
  int Count(int limit) {const auto count=Integer();if (count>static_cast<std::uint64_t>(limit)) throw std::runtime_error("Invalid snapshot count");return count;}
  double Double() {auto bits=Integer();double value;std::memcpy(&value,&bits,8);return value;}
  Vector3 Vector() {Vector3 v;for (int i=0;i<3;++i) v[i]=Double();return v;}
  State Controls() {
    State state;const int count=Count(512);
    for (int i=0;i<count;++i) {
      Eigen::Quaterniond q;for (int j=0;j<4;++j) q.coeffs()[j]=Double();
      if (!q.coeffs().allFinite() || std::abs(q.norm()-1)>1e-8) throw std::runtime_error("Invalid snapshot quaternion");
      state.rotations.emplace_back(q);state.positions.push_back(Vector());
    }
    state.gyro_bias=Vector();state.accel_bias=Vector();return state;
  }
  Eigen::MatrixXd Matrix() {
    const int rows=Count(3078),cols=Count(3078);
    if (static_cast<std::uint64_t>(rows)*cols>(bytes.size()-position)/8) throw std::runtime_error("Truncated snapshot matrix");
    Eigen::MatrixXd matrix(rows,cols);
    for (int r=0;r<rows;++r) for (int c=0;c<cols;++c) matrix(r,c)=Double();return matrix;
  }
};
std::string Serialize(const Snapshot& s) {
  Writer out;out.Integer(s.frame);out.Integer(s.outer_iteration);out.Double(s.start_time);out.Double(s.dt);out.Double(s.native_damping);
  out.Controls(s.initial);out.Vector(s.gravity);out.Vector(s.gyro_reference);out.Vector(s.accel_reference);
  for (int i=0;i<4;++i) out.Double(s.imu_weights[i]);
  out.Integer(s.fixed_knots.size());for (int i:s.fixed_knots) out.Integer(i);
  out.Integer(s.imu.size());for (const auto& o:s.imu) {out.Integer(o.span);out.Double(o.u);out.Double(o.timestamp);out.Vector(o.gyro);out.Vector(o.accel);}
  out.Integer(s.lidar.size());for (const auto& o:s.lidar) {out.Integer(o.span);out.Double(o.u);out.Double(o.timestamp);out.Vector(o.point);out.Vector(o.normal);out.Double(o.offset);out.Double(o.weight);}
  out.Integer(s.prior.knots.size());for (int i:s.prior.knots) out.Integer(i);
  out.Controls(s.prior.reference);out.Matrix(s.prior.sqrt_information);out.Matrix(s.prior.offset);
  return out.bytes;
}
}
std::string SnapshotDigest(const Snapshot& s) {return Hex(Hash(Serialize(s)));}
void SaveSnapshot(const Snapshot& s,const std::string& path) {
  s.Validate();const std::string payload=Serialize(s);
  Writer header;header.Integer(1);header.Integer(payload.size());header.Integer(Hash(payload));
  const auto parent=std::filesystem::path(path).parent_path();if (!parent.empty()) std::filesystem::create_directories(parent);
  const auto temporary=path+".tmp."+std::to_string(getpid());
  std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
  out.write("SLICTABC",8);out.write(header.bytes.data(),header.bytes.size());out.write(payload.data(),payload.size());
  out.close();if (!out) throw std::runtime_error("Cannot write snapshot: "+temporary);
  std::filesystem::rename(temporary,path);
}
Snapshot LoadSnapshot(const std::string& path) {
  const auto size=std::filesystem::file_size(path);
  if (size<32 || size>512ULL*1024*1024) throw std::runtime_error("Invalid snapshot file size");
  std::ifstream in(path,std::ios::binary);std::string bytes(size,'\0');
  in.read(bytes.data(),bytes.size());if (!in || bytes.compare(0,8,"SLICTABC")!=0) throw std::runtime_error("Invalid snapshot header");
  Reader header{bytes,8};if (header.Integer()!=1) throw std::runtime_error("Unsupported snapshot version");
  const auto payload_size=header.Integer(),hash=header.Integer();
  const auto payload=bytes.substr(32);
  if (payload_size!=payload.size() || Hash(payload)!=hash) throw std::runtime_error("Snapshot checksum mismatch");
  Reader r{payload};Snapshot s;s.frame=r.Integer();s.outer_iteration=r.Count(10000);
  s.start_time=r.Double();s.dt=r.Double();s.native_damping=r.Double();s.initial=r.Controls();
  s.gravity=r.Vector();s.gyro_reference=r.Vector();s.accel_reference=r.Vector();
  for (int i=0;i<4;++i) s.imu_weights[i]=r.Double();
  int count=r.Count(512);for (int i=0;i<count;++i) s.fixed_knots.push_back(r.Count(511));
  count=r.Count(1000000);for (int i=0;i<count;++i) {ImuObservation o;o.span=r.Count(511);o.u=r.Double();o.timestamp=r.Double();o.gyro=r.Vector();o.accel=r.Vector();s.imu.push_back(o);}
  count=r.Count(1000000);for (int i=0;i<count;++i) {LidarObservation o;o.span=r.Count(511);o.u=r.Double();o.timestamp=r.Double();o.point=r.Vector();o.normal=r.Vector();o.offset=r.Double();o.weight=r.Double();s.lidar.push_back(o);}
  count=r.Count(512);for (int i=0;i<count;++i) s.prior.knots.push_back(r.Count(511));
  s.prior.reference=r.Controls();s.prior.sqrt_information=r.Matrix();
  const auto offset=r.Matrix();if (offset.cols()!=1) throw std::runtime_error("Invalid prior offset shape");s.prior.offset=offset;
  if (r.position!=payload.size()) throw std::runtime_error("Trailing snapshot data");
  s.Validate();return s;
}
void WriteCsvHeader(std::ostream& out) {
  out<<"snapshot,frame,outer_iteration,backend,threads,iteration_limit,repeat,knots,imu_observations,lidar_observations,prior_rows,residual_blocks,scalar_residuals,reset_ms,build_ms,evaluate_ms,assemble_ms,linear_ms,update_ms,solve_ms,destroy_ms,total_ms,diagnostic_ms,initial_cost,final_cost,iterations,successful_steps,unsuccessful_steps,usable,termination,marginalize_ms,input_ms,process_peak_rss_kib,residual_evaluation_ms,jacobian_evaluation_ms,window_start_time,backend_with_diagnostics_ms,ceres_initial_trust_region_radius,ceres_final_trust_region_radius\n";
}
void WriteCsvRow(std::ostream& out,const Snapshot& s,const Options& o,const Metrics& m,int repeat,
                 const std::string& digest,double marginalize_ms,double input_ms) {
  struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
  out<<std::setprecision(17)<<digest<<','<<s.frame<<','<<s.outer_iteration<<','<<BackendName(o.backend)<<','<<o.threads<<','<<o.iterations<<','<<repeat<<','
     <<s.initial.rotations.size()<<','<<s.imu.size()<<','<<s.lidar.size()<<','<<s.prior.offset.size()<<','<<m.residual_blocks<<','<<m.scalar_residuals<<','
     <<m.reset_ms<<','<<m.build_ms<<','<<m.evaluate_ms<<','<<m.assemble_ms<<','<<m.linear_ms<<','<<m.update_ms<<','<<m.solve_ms<<','<<m.destroy_ms<<','<<m.total_ms<<','<<m.diagnostic_ms<<','
     <<m.initial_cost<<','<<m.final_cost<<','<<m.iterations<<','<<m.successful_steps<<','<<m.unsuccessful_steps<<','<<m.usable<<','<<m.termination<<','
     <<marginalize_ms<<','<<input_ms<<','<<usage.ru_maxrss<<','<<m.residual_evaluation_ms<<','<<m.jacobian_evaluation_ms<<','
     <<s.start_time<<','<<m.total_ms+m.diagnostic_ms+marginalize_ms+input_ms<<','
     <<(UsesCeresSolver(o.backend)?o.ceres_initial_trust_region_radius:0)<<','<<m.ceres_final_trust_region_radius<<'\n';
}
}  // namespace slict::comparison
