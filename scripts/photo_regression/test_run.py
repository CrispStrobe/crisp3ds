import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import json

spec=importlib.util.spec_from_file_location('photo_regression',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class GeometryTest(unittest.TestCase):
    def test_detects_coordinate_and_winding_changes(self):
        # The check must catch a visually similar but reversed/changed triangle.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            root=Path(root);stl=root/'mesh.stl';glb=root/'mesh.glb'
            vertices=struct.pack('<9f',0,0,0,1,0,0,0,1,0)
            stl.write_bytes(bytes(80)+struct.pack('<I',1)+bytes(12)+vertices+bytes(2))
            doc={'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':1}]}],'accessors':[{'bufferView':0,'componentType':5126,'count':3},{'bufferView':1,'componentType':5125,'count':3}],'bufferViews':[{'byteOffset':0},{'byteOffset':36}]}
            text=json.dumps(doc).encode();text+=b' '*((-len(text))%4)
            def write(indices):
                binary=vertices+struct.pack('<3I',*indices)
                glb.write_bytes(b'glTF'+struct.pack('<II',2,28+len(text)+len(binary))+struct.pack('<II',len(text),0x4e4f534a)+text+struct.pack('<II',len(binary),0x004e4942)+binary)
            write([0,1,2]);self.assertTrue(runner.verify_geometry(stl,glb)['triangle_coordinates_and_winding_byte_equal'])
            write([0,2,1])
            with self.assertRaisesRegex(ValueError,'geometry differs'):runner.verify_geometry(stl,glb)

if __name__=='__main__':unittest.main()
