#include "crisp3ds/core.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::pair<crisp3ds_status,std::string> run(const std::string &s, bool reconstruct=false) {
  auto fn = reconstruct ? crisp3ds_reconstruct_json : crisp3ds_validate_project_json;
  size_t needed=0;
  assert(fn(s.data(),s.size(),nullptr,0,&needed)==CRISP3DS_BUFFER_TOO_SMALL);
  std::vector<char> out(needed);
  auto status=fn(s.data(),s.size(),out.data(),out.size(),&needed);
  return {status,std::string(out.data())};
}
void check_invalid(const std::string &s, const std::string &fragment) {
  auto [status,json]=run(s);
  if (status != CRISP3DS_INVALID_PROJECT || json.find(fragment)==std::string::npos) {
    std::cerr << "Expected invalid project containing '" << fragment << "', got: " << json << '\n';
    std::abort();
  }
}
}

int main() {
  assert(crisp3ds_abi_version()==1);
  const std::string empty=R"({"schemaVersion":1,"id":"scan-1","name":"Test","units":"mm","images":[],"stages":{}})";
  auto [status,valid]=run(empty);
  assert(status==CRISP3DS_OK && valid.find("\"imageCount\":0")!=std::string::npos);
  auto [reconstruction,unavailable]=run(empty,true);
  assert(reconstruction==CRISP3DS_UNAVAILABLE && unavailable.find("no mesh was produced")!=std::string::npos);
  check_invalid(R"({"schemaVersion":2,"id":"x","name":"x","units":"mm","images":[],"stages":{}})","schemaVersion");
  check_invalid(R"({"schemaVersion":1,"id":"x","name":"x","units":"mm","images":[{"id":"a","path":"../secret.jpg"}],"stages":{}})","unsafe path segment");
  check_invalid(R"({"schemaVersion":1,"id":"x","name":"x","units":"mm","images":[{"id":"a","path":"C:\\x.jpg"}],"stages":{}})","relative / separators");
  check_invalid(R"({"schemaVersion":1,"id":"x","name":"x","units":"mm","images":[{"id":"a","path":"a.jpg"},{"id":"a","path":"b.jpg"}],"stages":{}})","unique");
  check_invalid(R"({"schemaVersion":1,"id":"x","name":"x","units":"mm","images":[],"board":{"type":"aruco","markerSizeMm":20,"rotatesWithObject":false},"stages":{}})","rotatesWithObject");
  check_invalid(R"({"schemaVersion":1,"id":"x","name":"x","units":"mm","images":[],"stages":{"dense":"complete","dense":"failed"}})","duplicate object key");
  check_invalid(R"({"schemaVersion":1,"id":"x","name":"x","units":"mm","images":[],"stages":{"dense":"done"}})","invalid stage state");
  check_invalid(R"({"schemaVersion":1,"id":"x","name":"x","units":"mm","images":[],"stages":{},"calibration":{"width":0,"height":480,"fx":100,"fy":100,"cx":320,"cy":240,"distortion":[]}})","calibration.width");
  check_invalid(R"({"schemaVersion":1,"id":"x","name":"   ","units":"mm","images":[],"stages":{}})","whitespace only");
  check_invalid(R"({"schemaVersion":1,"id":"x","name":"\u00a0","units":"mm","images":[],"stages":{}})","whitespace only");
  check_invalid(R"({"schemaVersion":1,"id":"x","name":"x","units":"mm","images":[{"id":"a","path":" images/a.jpg"}],"stages":{}})","edge whitespace");
  std::string malformed_utf8=empty;
  malformed_utf8.replace(malformed_utf8.find("Test"),4,std::string(1,static_cast<char>(0xff)));
  check_invalid(malformed_utf8,"UTF-8");
  size_t required=0;
  assert(crisp3ds_geometry_diagnostic_json(nullptr,0,&required)==CRISP3DS_BUFFER_TOO_SMALL);
  std::vector<char> short_buffer(required, '#');
  assert(crisp3ds_geometry_diagnostic_json(short_buffer.data(),required-1,&required)==CRISP3DS_BUFFER_TOO_SMALL);
  assert(short_buffer.front()=='#');
  std::vector<char> diagnostic(required);
  assert(crisp3ds_geometry_diagnostic_json(diagnostic.data(),diagnostic.size(),&required)==CRISP3DS_OK);
  assert(std::strstr(diagnostic.data(),"\"ok\":true")!=nullptr);
  assert(crisp3ds_capabilities_json(nullptr,0,nullptr)==CRISP3DS_INVALID_ARGUMENT);
  char *sparse_report=nullptr;
  size_t sparse_size=0;
  assert(crisp3ds_reconstruct_sparse_alloc(nullptr,&sparse_report,&sparse_size)==CRISP3DS_INVALID_ARGUMENT);
  assert(crisp3ds_reconstruct_sparse_alloc("",&sparse_report,&sparse_size)==CRISP3DS_INVALID_ARGUMENT);
#ifndef CRISP3DS_WITH_OPENCV
  assert(crisp3ds_reconstruct_sparse_alloc("missing-project.json",&sparse_report,&sparse_size)==CRISP3DS_UNAVAILABLE);
  assert(sparse_report && sparse_size==std::strlen(sparse_report)+1);
  assert(std::strstr(sparse_report,"\"ok\":false")!=nullptr);
  crisp3ds_free(sparse_report);
#endif
  std::cout << "core contract tests passed\n";
}
