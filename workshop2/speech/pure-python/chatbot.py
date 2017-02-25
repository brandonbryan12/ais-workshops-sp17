import neural_net as nn
import corpus_parser as corpus
import performAction as bot


if __name__ == '__main__':
    nn.train()
    print('Conversation Started!')
    # main loop
    done = False
    while(not done):
        query = input(">>> ")
        if query == 'exit':
            done = True
        else:
            res = nn.response(query)
            print(bot.performBotAction(res))        
