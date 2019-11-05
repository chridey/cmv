import json
import socket
import select
import time

class TimeoutException(Exception):
    pass

class SemaforParser:
    def get_frames(self, conll_string):
        frames = self.parse(conll_string)
        
        return [SemaforAnnotation(**frame) for frame in frames]

class TCPClientSemaforParser(SemaforParser):
    def __init__(self, host='localhost', port=8998):
        self.host = host
        self.port = port

    def parse(self, conll_string):
        sock = socket.create_connection((self.host, self.port))
        sock.setblocking(0)
        sock.sendall(conll_string.encode('utf-8'))
        #sleep?
        sock.shutdown(socket.SHUT_WR)
        frame_string = ''
        start = time.time()
        timeout_in_seconds = 60
        while 1:
            if time.time() - start > timeout_in_seconds:
                raise TimeoutException

            ready = select.select([sock], [], [], timeout_in_seconds)
            if ready[0]:
                data = sock.recv(1024)
            else:
                raise TimeoutException
                   
            #data = sock.recv(1024)
            if data == "":
                break
            frame_string += data
        ret = []
        tokens = [i.split('\n') for i in conll_string.split('\n\n')]
        
        for index,line in enumerate(frame_string.splitlines()):
            frames = json.loads(line)
            if type(frames) == dict:
                ret.append(frames)
                continue
            frames_new = {'tokens': [i.split('\t')[1] for i in tokens[index]],
                          'frames': []}
            for frame in frames:
                spans = []
                span = {}
                for location in sorted(frame['first']):
                    if not len(span):
                        span['start'] = location
                        span['end'] = location+1
                        span['text'] = frames_new['tokens'][location]
                    else:
                        if location > span['end']:
                            spans.append(span)
                            span = {'start': location,
                                    'end': location+1,
                                    'text': frames_new['tokens'][location]}
                        else:
                            span['end'] = location+1
                            span['text'] += ' ' + frames_new['tokens'][location]
                    
                spans.append(span)
                frames_new['frames'].append({'target': {'name': frame['second'],
                                                        'spans': spans}})
            ret.append(frames_new)
        return ret

class SemaforAnnotation:
    def __init__(self, frames=None, tokens=None, error=None):
        self.frames = []
        if frames is not None:
            self.frames = frames
            
        self.tokens = []
        if tokens is not None:
            self.tokens = tokens
            
        self.error = error
        
    def iterTargets(self):
        for frame in self.frames:
            target = frame['target']
            frame_name = target['name']
            for span in target['spans']:
                for r in range(span['start'], span['end']):
                    yield frame_name,r
