from PIL import Image
import os.path
import shutil  # 用来删除文件夹

#把图片转换为指定的尺寸的RGB或灰度图片
def convert_image(path,out_dir,size,color='RGB'):
    img=Image.open(path)
    if color=='L':
        img = img.convert('L')
    try:
        new_img=img.resize((size,size),Image.BILINEAR)
        new_img.save(os.path.join(out_dir,os.path.basename(path)))
    except Exception as e:
        #处理图片存储不同模式的问题
        if img.mode in ['P','RGBA']:
            img = img.convert('RGB')
            new_img=img.resize((size,size),Image.BILINEAR)
            new_img.save(os.path.join(out_dir,os.path.basename(path)))
            print(path,'Model change success')
        else:
            print(e,path)

#转换，参数见注释
if __name__=='__main__':

    path = './collections' 
    classes = os.listdir(path) 
    print(f'Total number of dir: {len(classes)}')

    #定义要转换的尺寸，宽高相同
    width = 227;
    #定义输出的颜色，RGB或L（黑白）
    color = 'RGB'
    #建立输出文件夹，命名方式为颜色+尺寸
    width_path = './Standard/'+color+str(width) 
    if os.path.exists(width_path):
        shutil.rmtree(width_path)
    os.mkdir(width_path)
    #遍历图片，开始转换
    for c in classes:
        output = os.path.join(width_path,c)
        if os.path.exists(output):
            shutil.rmtree(output)
        os.mkdir(output)
        if c == '.DS_Store':
                continue
        dir = os.listdir(os.path.join(path, c))
        for item in dir:
            if item == '.DS_Store':
                continue
            img_dir = os.path.join(path,c,item)
            convert_image(img_dir,output,width,color)
            
