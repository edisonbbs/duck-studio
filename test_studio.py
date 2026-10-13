import json, tempfile, unittest, pathlib, sys, unittest.mock as mock
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import versions, server
class StudioTests(unittest.TestCase):
    def test_paths_and_input(self):
        with self.assertRaises(ValueError):server.log_path('../../.bashrc')
        with self.assertRaises(ValueError):versions.get_version('../v001')
    def test_version_recipe(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(versions,'STORE',pathlib.Path(temp)):
                versions.init_baseline();base=versions.get_version('v001');before=versions.sha(versions.REPO/versions.CONST)
                v=versions.create_version({'parent':'v001','name':'test','params':{'home_deg':23,'kp':6,'com_x_mm':-7,'root_z_mm':0}})
                self.assertEqual(v['params']['home_deg'],23);self.assertEqual(v['params']['kp'],6)
                self.assertEqual(versions.sha(versions.REPO/versions.CONST),before)
                self.assertEqual(versions.get_version('v001')['params'],base['params'])
                self.assertGreater(v['params']['stand_z_mm'],100)
                import xml.etree.ElementTree as ET
                for file in ['robot_walk.xml','robot_allcollisions.xml']:
                    i=ET.parse(versions.STORE/v['id']/'workspace'/versions.ROBOT/file).find('.//body[@name="trunk_base"]/inertial')
                    self.assertEqual(float(i.attrib['pos'].split()[0]),-.007)
                versions.verify(v['id'])
                root=versions.STORE/v['id']/'workspace'
                (root/'logs').mkdir();(root/'logs'/'new.log').write_text('new')
                versions.verify(v['id'])
                (root/versions.CONST).write_text((root/versions.CONST).read_text()+'\n# changed')
                with self.assertRaises(ValueError):versions.verify(v['id'])
    def test_no_double_train(self):
        with mock.patch.object(server,'TASKS',{'test':'xgoduck_velocity'}), mock.patch.object(server,'external_processes',return_value=[{'kind':'train'}]),mock.patch.object(versions,'get_version',return_value={'kind':'recipe'}),mock.patch.object(versions,'verify'):
            with self.assertRaisesRegex(ValueError,'已有訓練'):server.start('train',{'task':'test','version':'v002'})
if __name__=='__main__':unittest.main()