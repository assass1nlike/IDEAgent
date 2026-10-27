import json
import hashlib
from collections import Counter
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import httpx
from openai import OpenAI
from ideagent.clients import OpenAIClient
from ideagent.request_checkpoint import RequestJournal, configure, prepare_checkpoint


def response(text):
    return httpx.Response(200, json={
        'id': text, 'object': 'chat.completion', 'created': 0, 'model': 'test-model',
        'choices': [{'index': 0, 'finish_reason': 'stop',
                     'message': {'role': 'assistant', 'content': text}}],
        'usage': {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7}})


def client(journal, handler):
    configure(journal)
    with patch.dict(os.environ, TEST_API_KEY='unused'):
        c = OpenAIClient(model_id='gpt-test', api_key_env='TEST_API_KEY', timeout=120)
    c.client.close()
    c.client = OpenAI(api_key='unused', base_url='https://example.test/v1', max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return c


def subprocess_workflow(directory, resume):
    root = Path(directory)
    journal = RequestJournal(root/'requests.sqlite', {'source': 'same', 'max_tokens': 32000}, resume=resume)
    def handler(request):
        prompt = json.loads(request.content)['messages'][0]['content']
        with (root/'network.jsonl').open('a') as f:
            f.write(json.dumps(prompt)+'\n'); f.flush(); os.fsync(f.fileno())
        if prompt == 'interrupt' and not resume:
            os._exit(71)
        return response('sample-' + str(len((root/'network.jsonl').read_text().splitlines())))
    c = client(journal, handler)
    values = []
    with c.client:
        for prompt in ['same', 'same', 'interrupt']:
            values.append(c.generate([{'role': 'user', 'content': prompt}], max_new_tokens=32000))
    (root/'result.json').write_text(json.dumps(values))
    journal.close()


def native_workflow(directory, resume, interrupt):
    from ideagent.agents import IdeatorAgent, CriticAgent, QualityAgent, FEEDBACK_RUBRIC_IDS
    from ideagent.generation_loop import Config, GenerationLoop
    from ideagent.signature import SIGNATURE_FIELDS
    root=Path(directory)
    journal=RequestJournal(root/'requests.sqlite', {'native_test':1},resume=resume)
    out=root/('resumed' if resume else 'first');out.mkdir()
    network_lock=threading.Lock()
    quality_count=0
    def make(role):
        def handler(request):
            nonlocal quality_count
            payload=json.loads(request.content)
            encoded=json.dumps(payload,sort_keys=True)
            fingerprint=hashlib.sha256(encoded.encode()).hexdigest()
            schema=payload.get('response_format',{}).get('json_schema',{}).get('name','text')
            with network_lock:
                with (root/'network.jsonl').open('a') as f:
                    f.write(json.dumps({'role':role,'request':fingerprint,'schema':schema})+'\n')
                    f.flush();os.fsync(f.fileno())
                if schema=='agentic_feedback':
                    quality_count+=1
                    if interrupt and quality_count==2:
                        os._exit(72)
            if schema=='text':
                text='Research idea '+fingerprint[:12]
            elif schema=='signature':
                text=json.dumps({name:name+' '+fingerprint[:12] for name in SIGNATURE_FIELDS})
            elif schema=='agentic_feedback':
                text=json.dumps({'rubric_assessment':[{'id':rid,'score':100,'evidence':'verified','critic_pressure':''} for rid in FEEDBACK_RUBRIC_IDS]})
            elif schema=='diversity_verdict':
                text=json.dumps({'closest':[],'duplicate_ids':[],'historical_duplicate_ids':[],
                                 'diversity_score':100,'matches_rejected_family':False,'rejected_family':''})
            else:
                text=json.dumps({'score':9,'rationale':'valid'})
            return response(text)
        c=client(journal,handler);c.checkpoint_role=role;return c
    clients={role:make(role) for role in ['ideator','critic','quality','steno','judge']}
    loop=GenerationLoop(config=Config(input_jsonl='unused',output_dir=str(out),max_budget=2),
        ideator=IdeatorAgent(role='ideator',client=clients['ideator'],gen_kwargs={}),
        critic=CriticAgent(role='critic',client=clients['critic'],gen_kwargs={}),
        quality=QualityAgent(role='quality',client=clients['quality'],gen_kwargs={}),
        steno_client=clients['steno'],steno_gen_kwargs={},judge_client=clients['judge'],judge_gen_kwargs={})
    archive=loop.run_topic(topic_id='test',background='Fixed paper background',out_dir=out)
    (root/'result.json').write_text(json.dumps(archive.to_json(),sort_keys=True))
    for c in clients.values():c.client.close()
    journal.close()


class CheckpointTest(unittest.TestCase):
    def tearDown(self):
        configure(None)

    def test_new_process_replays_repeated_successes_and_resends_interrupted_request(self):
        with TemporaryDirectory() as root:
            script = 'from tests.test_checkpoint import subprocess_workflow; import sys; subprocess_workflow(sys.argv[1],sys.argv[2]=="1")'
            first = subprocess.run([sys.executable, '-c', script, root, '0'], capture_output=True)
            self.assertEqual(first.returncode, 71, first.stderr)
            second = subprocess.run([sys.executable, '-c', script, root, '1'], capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(json.loads((Path(root)/'result.json').read_text()), ['sample-1','sample-2','sample-4'])
            self.assertEqual([json.loads(l) for l in (Path(root)/'network.jsonl').read_text().splitlines()], ['same','same','interrupt','interrupt'])

    def test_parallel_slots_survive_reversed_scheduling(self):
        with TemporaryDirectory() as root:
            path=Path(root)/'requests.sqlite'; j=RequestJournal(path,{})
            calls=[]; lock=threading.Lock()
            def handler(request):
                with lock:
                    calls.append(1); number=len(calls)
                return response('answer-'+str(number))
            c=client(j,handler)
            def run(i): return c.generate([{'role':'user','content':'same'}], _checkpoint_slot=['soundness',i])
            with c.client, ThreadPoolExecutor(3) as pool:
                expected=dict(zip(range(3),pool.map(run,range(3))))
            j.close(); j=RequestJournal(path,{},resume=True)
            c=client(j,lambda _: self.fail('Replayed parallel response must not call API'))
            with c.client, ThreadPoolExecutor(3) as pool:
                actual=dict(zip([2,1,0],pool.map(run,[2,1,0])))
            self.assertEqual(actual,expected);j.close()

    def test_native_search_reconstructs_candidate_archive_after_process_interruption(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);baseline=root/'baseline';interrupted=root/'interrupted'
            baseline.mkdir();interrupted.mkdir()
            code='from tests.test_checkpoint import native_workflow; import sys; native_workflow(sys.argv[1],sys.argv[2]=="1",sys.argv[3]=="1")'
            def run(path,resume,interrupt):
                return subprocess.run([sys.executable,'-c',code,str(path),str(int(resume)),str(int(interrupt))],capture_output=True,text=True)
            full=run(baseline,False,False);self.assertEqual(full.returncode,0,full.stderr)
            first=run(interrupted,False,True);self.assertEqual(first.returncode,72,first.stderr)
            first_candidates=(interrupted/'first/candidates_test.jsonl').read_text().splitlines()
            self.assertEqual(len(first_candidates),1)
            second=run(interrupted,True,False);self.assertEqual(second.returncode,0,second.stderr)
            self.assertEqual(json.loads((baseline/'result.json').read_text()),json.loads((interrupted/'result.json').read_text()))
            self.assertEqual((baseline/'first/candidates_test.jsonl').read_text(),(interrupted/'resumed/candidates_test.jsonl').read_text())
            calls=lambda path:[json.loads(line) for line in (path/'network.jsonl').read_text().splitlines()]
            expected=Counter((r['role'],r['request']) for r in calls(baseline))
            actual=Counter((r['role'],r['request']) for r in calls(interrupted))
            extra=actual-expected
            self.assertEqual(sum(extra.values()),1)
            self.assertEqual(next(iter(extra))[0],'quality')
            self.assertFalse(expected-actual)

    def test_attempts_preserved_and_changed_configuration_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / 'input.jsonl'
            input_file.write_text('{"topic_id": "test"}\n')
            config = {'data': {'input_jsonl': str(input_file), 'output_dir': 'unused'},
                      'run': {'max_budget': 2, 'seed': 17}}
            journal, first = prepare_checkpoint(config, root / 'run')
            journal.close()
            journal, second = prepare_checkpoint(config, root / 'run', resume=True)
            journal.close()
            self.assertNotEqual(first['data']['output_dir'], second['data']['output_dir'])
            self.assertTrue(Path(first['data']['output_dir'], 'config.json').exists())
            config['run']['max_budget'] = 3
            with self.assertRaisesRegex(ValueError, 'Resume configuration differs'):
                prepare_checkpoint(config, root / 'run', resume=True)
            self.assertFalse((root / 'run/attempts/0003').exists())

    def test_changed_parameters_rejected_and_active_run_locked(self):
        with TemporaryDirectory() as root:
            path=Path(root)/'requests.sqlite';j=RequestJournal(path,{'max_tokens':32000})
            with self.assertRaises(RuntimeError): RequestJournal(path,{'max_tokens':32000},resume=True)
            j.close()
            with self.assertRaises(ValueError): RequestJournal(path,{'max_tokens':64000},resume=True)
            j=RequestJournal(path,{'max_tokens':32000},resume=True);j.close()




def raw_response_workflow(directory, resume):
    root = Path(directory)
    journal = RequestJournal(root / 'requests.sqlite', {}, resume=resume)
    def handler(request):
        with (root / 'network').open('a') as f:
            f.write('request\n')
        return response('saved before parsing')
    if not resume:
        original = journal.save_response
        def crash(*args):
            original(*args)
            os._exit(73)
        journal.save_response = crash
    c = client(journal, handler)
    with c.client:
        text = c.generate([{'role': 'user', 'content': 'raw'}])
    (root / 'result').write_text(text)
    journal.close()


class ProviderCheckpointTest(unittest.TestCase):
    def tearDown(self):
        configure(None)

    def test_response_committed_before_extraction_survives_process_exit(self):
        with TemporaryDirectory() as root:
            code = 'from tests.test_checkpoint import raw_response_workflow; import sys; raw_response_workflow(sys.argv[1], sys.argv[2]=="1")'
            first = subprocess.run([sys.executable, '-c', code, root, '0'], capture_output=True)
            self.assertEqual(first.returncode, 73, first.stderr)
            second = subprocess.run([sys.executable, '-c', code, root, '1'], capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual((Path(root) / 'network').read_text(), 'request\n')
            self.assertEqual((Path(root) / 'result').read_text(), 'saved before parsing')

    def test_gemini_and_anthropic_sync_async_and_fanout_replay(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import Mock, AsyncMock
        from google.genai import types
        from anthropic.types import Message, TextBlock, Usage
        from ideagent.clients import GeminiClient, AnthropicClient
        providers = [
            (GeminiClient, types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(parts=[types.Part(text='answer')]), finish_reason='STOP')])),
            (AnthropicClient, Message(id='msg', type='message', role='assistant', model='claude-test',
                content=[TextBlock(type='text', text='answer')], stop_reason='end_turn',
                usage=Usage(input_tokens=1, output_tokens=1))),
        ]
        for cls, value in providers:
            with self.subTest(provider=cls.__name__), TemporaryDirectory() as root:
                for resume in (False, True):
                    journal = RequestJournal(Path(root) / 'requests.sqlite', {}, resume=resume)
                    configure(journal)
                    c = cls.__new__(cls)
                    c.model_id = 'test-model'
                    c._init_response_metadata()
                    sync = Mock(return_value=value)
                    async_ = AsyncMock(return_value=value)
                    c.client = SimpleNamespace(
                        models=SimpleNamespace(generate_content=sync),
                        aio=SimpleNamespace(models=SimpleNamespace(generate_content=async_)),
                        messages=SimpleNamespace(create=sync))
                    c.async_client = SimpleNamespace(messages=SimpleNamespace(create=async_))
                    messages = [{'role': 'user', 'content': 'question'}]
                    self.assertEqual(c.generate(messages), 'answer')
                    self.assertEqual(asyncio.run(c.generate_async(messages)), 'answer')
                    self.assertEqual(c.generate_many(messages, n=3), ['answer'] * 3)
                    self.assertEqual(sync.call_count, 0 if resume else 4)
                    self.assertEqual(async_.call_count, 0 if resume else 1)
                    journal.close()

    def test_vllm_dict_response_and_native_multi_candidate_replay(self):
        from unittest.mock import Mock
        from ideagent.clients import VLLMOpenAIClient
        with TemporaryDirectory() as root:
            for resume in (False, True):
                journal = RequestJournal(Path(root) / 'requests.sqlite', {}, resume=resume)
                configure(journal)
                c = VLLMOpenAIClient.__new__(VLLMOpenAIClient)
                c.model_id = 'local-model'
                c._init_response_metadata()
                send = Mock(return_value={'choices': [
                    {'message': {'content': 'first'}, 'finish_reason': 'stop'},
                    {'message': {'content': 'second'}, 'finish_reason': 'stop'}]})
                c._chat_completion_create = send
                self.assertEqual(c.generate_many([{'role': 'user', 'content': 'question'}], n=2), ['first', 'second'])
                self.assertEqual(send.call_count, 0 if resume else 1)
                if not resume:
                    # Force replay through raw-response reconstruction rather than final text.
                    journal.db.execute('UPDATE calls SET response=NULL')
                    journal.db.commit()
                journal.close()

    def test_native_cli_replays_into_new_attempt_and_checks_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / 'input.jsonl'
            input_file.write_text('')
            config = {'data': {'input_jsonl': str(input_file), 'output_dir': 'unused'},
                      'run': {'max_budget': 1, 'seed': 17},
                      'agents': {role: {'model_id': 'gpt-test', 'api_key_env': 'TEST_API_KEY'}
                                 for role in ('ideator', 'critic', 'quality', 'steno', 'judge')}}
            config_file = root / 'config.yaml'
            config_file.write_text(json.dumps(config))
            command = [sys.executable, 'scripts/run_ideagent.py', '--config', str(config_file),
                       '--checkpoint-dir', str(root / 'run')]
            for suffix in ([], ['--resume']):
                result = subprocess.run(command + suffix, env={**os.environ, 'TEST_API_KEY': 'unused'},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / 'run/attempts/0001/final_idea_cores.jsonl').exists())
            self.assertTrue((root / 'run/attempts/0002/final_idea_cores.jsonl').exists())
            input_file.write_text('changed input')
            result = subprocess.run(command + ['--resume'], env={**os.environ, 'TEST_API_KEY': 'unused'},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Resume configuration differs', result.stderr)

    def test_no_checkpoint_keeps_independent_sampling(self):
        configure(None)
        calls = []
        def handler(request):
            calls.append(1)
            return response(str(len(calls)))
        c = client(None, handler)
        with c.client:
            messages = [{'role': 'user', 'content': 'same'}]
            self.assertEqual(c.generate(messages), '1')
            self.assertEqual(c.generate(messages), '2')


if __name__ == "__main__":
    unittest.main()
