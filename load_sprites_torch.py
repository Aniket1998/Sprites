import os
import numpy as np
import torch
import torchvision.transforms as T

transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

directions = ('front', 'left', 'right')
actions = ('walk', 'spellcard', 'slash')
path = './npy/'
id_train = 0
id_test = 0
for act in range(len(actions)):
    for dir in range(len(directions)):
        # Prepare the Training Dataset
        x = np.load(path + '%s_%s_frames_train.npy' % (actions[act], directions[dir]))
        a = np.load(path + '%s_%s_attributes_train.npy' % (actions[act], directions[dir]))
        for i in range(x.shape[0]):
            # Extract the video frames
            video = x[i]
            labels = a[i]
            frames = []
            for j in range(video.shape[0]):
                frames.append(transform(video[j]))
            frames = torch.stack(frames)
            #Extract the labels
            body = np.nonzero(labels[0][0])[0].item()
            bottom = np.nonzero(labels[1][0])[0].item()
            top = np.nonzero(labels[2][0])[0].item()
            hair = np.nonzero(labels[3][0])[0].item()
            train_dict = {
                    'frames': frames,
                    'id': id_train,
                    'body': body,
                    'bottom': bottom,
                    'top': top,
                    'hair': hair,
                    'action': act,
                    'orientation': dir
            }
            torch.save(train_dict, './train_sprites/%d.sprite' % id_train)
            id_train += 1
        # Test Set
        x = np.load(path + '%s_%s_frames_test.npy' % (actions[act], directions[dir]))
        a = np.load(path + '%s_%s_attributes_test.npy' % (actions[act], directions[dir]))
        print(x.shape[0])
        for i in range(x.shape[0]):
            # Extract the video frames
            video = x[i]
            labels = a[i]
            frames = []
            for j in range(video.shape[0]):
                frames.append(transform(video[j]))
            frames = torch.stack(frames)
            #Extract the labels
            body = np.nonzero(labels[0][0])[0].item()
            bottom = np.nonzero(labels[1][0])[0].item()
            top = np.nonzero(labels[2][0])[0].item()
            hair = np.nonzero(labels[3][0])[0].item()
            test_dict = {
                    'frames': frames,
                    'id': id_test,
                    'body': body,
                    'bottom': bottom,
                    'top': top,
                    'hair': hair,
                    'action': act,
                    'orientation': dir
            }
            torch.save(test_dict, './test_sprites/%d.sprite' % id_test)
            id_test += 1


print('Training Dataset %d' % id_train)
print('Test Dataset %d' % id_test)
