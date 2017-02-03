import shutil
import os

#start
#bad file

#suffix

import sys

start = int(sys.argv[1])
end = int(sys.argv[2])
suffix = sys.argv[3]
base_dir = sys.argv[4]

discourse_dir = base_dir #os.path.join(base_dir, 'discourse')
done_dir = os.path.join(discourse_dir, 'done')
bad_dir = os.path.join(discourse_dir, 'bad')
discourse_dir_suffix = os.path.join(discourse_dir, suffix)
done_dir_suffix = os.path.join(done_dir, suffix)
bad_dir_suffix = os.path.join(bad_dir, suffix)

print(discourse_dir_suffix, done_dir_suffix, bad_dir_suffix)

for filename in range(start, end):
    
    if os.path.exists(os.path.join(discourse_dir_suffix, str(filename))):
        if not os.path.exists(done_dir_suffix):
            os.makedirs(done_dir_suffix)
        shutil.move(os.path.join(discourse_dir_suffix, str(filename)),
                    os.path.join(done_dir_suffix, str(filename)))
    else:
        print(filename)
        
if os.path.exists(os.path.join(discourse_dir_suffix, str(end))):
    if not os.path.exists(bad_dir_suffix):
        os.makedirs(bad_dir_suffix)
    
    shutil.move(os.path.join(discourse_dir_suffix, str(end)),
                os.path.join(bad_dir_suffix, str(end)))
else:
    print(end)



