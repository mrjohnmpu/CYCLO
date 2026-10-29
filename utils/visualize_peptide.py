# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem import AllChem

# def draw_cyclic_peptides_grid(csv_path, output_image="cyclic_peptides_grid.png", max_mols=20):
#     # 1. 读取数据
#     try:
#         df = pd.read_csv(csv_path)
#         # 确保列名正确，如果不是 'SMILES' 请修改
#         smiles_list = df['SMILES'].tolist()
#     except Exception as e:
#         print(f"读取CSV失败: {e}")
#         return

#     mols = []
#     legends = []
    
#     print(f"正在处理前 {max_mols} 个分子...")
    
#     # 2. 转换为 RDKit 分子对象
#     for i, smi in enumerate(smiles_list[:max_mols]):
#         mol = Chem.MolFromSmiles(smi)
#         if mol:
#             # 生成 2D 坐标，这对于环肽展示非常重要
#             AllChem.Compute2DCoords(mol)
#             mols.append(mol)
#             legends.append(f"Seq_{i}")

#     # 3. 绘制网格图
#     if mols:
#         img = Draw.MolsToGridImage(
#             mols, 
#             molsPerRow=4,              # 每行显示几个
#             subImgSize=(400, 400),     # 每个小图的尺寸 (环肽比较大，建议设置大一点)
#             legends=legends,           # 显示每个分子的标签
#             useSVG=False               # 设置为 True 可以保存为矢量图
#         )
        
#         # 保存图片
#         img.save(output_image)
#         print(f"成功保存网格图至: {output_image}")
#     else:
#         print("没有有效的分子可以绘制。")

# # 使用示例
# # 请将 'clean_generated.csv' 替换为您清洗后保存的文件名
# draw_cyclic_peptides_grid('utils/clean_generated.csv', max_mols=16)


# import os
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem import AllChem

# def highlight_and_save_individual(csv_path, output_dir="output/generate/prompt_generate/peptide_images", max_mols=10):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     df = pd.read_csv(csv_path)
    
#     print(f"开始绘制并高亮环结构，输出目录: {output_dir}")

#     for i, smi in enumerate(df['SMILES'][:max_mols]):
#         mol = Chem.MolFromSmiles(smi)
#         if not mol:
#             continue
            
#         # 1. 计算 2D 坐标
#         AllChem.Compute2DCoords(mol)
        
#         # 2. 寻找最大环 (Macrocycle) 进行高亮
#         # 获取最小环集 (SSSR)
#         sssr = Chem.GetSymmSSSR(mol)
        
#         # 找到原子数大于等于 10 的大环 (环肽通常很大)
#         macrocycle_atoms = set()
#         for ring in sssr:
#             if len(ring) >= 10: 
#                 macrocycle_atoms.update(ring)
        
#         # 将集合转为列表
#         highlight_atoms = list(macrocycle_atoms)
        
#         # 3. 绘图配置
#         d = Draw.rdMolDraw2D.MolDraw2DSVG(500, 500) # 使用 Cairo 引擎绘制高质量 PNG
        
#         # 设置绘图选项
#         opts = d.drawOptions()
#         opts.addAtomIndices = False # 不显示原子索引，保持图片干净
        
#         # 绘制
#         d.DrawMolecule(mol, highlightAtoms=highlight_atoms)
#         d.FinishDrawing()

#         svg_text = d.GetDrawingText()
        
#         # 保存
#         output_file = os.path.join(output_dir, f"peptide_{i}_highlighted.svg")
#         # d.WriteDrawingText(output_file)
#         with open(output_file, 'w') as f:
#             f.write(svg_text)
        
#     print("绘制完成！")

# # 使用示例
# highlight_and_save_individual('utils/clean_generated.csv', max_mols=1)


#############################################################################
# 这部分是只有绘图部分
# import os
# import numpy as np
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import Draw
# from rdkit import DataStructs


# def get_canonical_smiles(smi):
#     """辅助函数：将 SMILES 转换为标准形式，用于去重对比"""
#     if not smi: return None
#     mol = Chem.MolFromSmiles(smi)
#     if mol:
#         return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) # 忽略立体异构可根据需求改为 True
#     return None

# def draw_highlighted_grid_svg(csv_path, output_dir, output_file="cyclic_peptides_grid_highlighted.svg", mols_per_row=4, max_mols=20):
#     """
#     在一张 SVG 网格图中绘制多个分子，并自动高亮检测到的大环结构。
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     # 1. 读取数据
#     try:
#         df = pd.read_csv(csv_path)
#         # smiles_list = df['SMILES'].tolist()
#         # 有的版本增加了is_valid列，需要读取合法的分子
#         smiles_list = df[df['is_valid'] == True]['smiles'].tolist()
#     except Exception as e:
#         print(f"读取文件失败: {e}")
#         return

#     mols = []
#     highlight_lists = []  # 这是一个列表的列表，存储每个分子需要高亮的原子索引
#     legends = []

#     print(f"正在处理前 {min(len(smiles_list), max_mols)} 个分子...")

#     # 2. 处理每个分子
#     for i, smi in enumerate(smiles_list[:max_mols]):
#         mol = Chem.MolFromSmiles(smi)
#         if not mol:
#             continue

#         # A. 生成 2D 坐标 (对于环肽非常重要，防止结构重叠)
#         AllChem.Compute2DCoords(mol)

#         # B. 寻找大环 (Macrocycle)
#         # 获取最小环集 (SSSR)
#         sssr = Chem.GetSymmSSSR(mol)
        
#         current_mol_highlights = set()
#         for ring in sssr:
#             # 阈值设为 10 或 12，视您的肽大小而定
#             if len(ring) >= 10: 
#                 current_mol_highlights.update(ring)
        
#         # 将集合转为列表加入总的高亮列表
#         # 注意：即使没有找到大环，也要加入一个空列表 []，以保持索引对齐
#         highlight_lists.append(list(current_mol_highlights))
        
#         # C. 添加到分子列表
#         mols.append(mol)
#         legends.append(f"Peptide_{i}")

#     if not mols:
#         print("没有有效的分子可绘制。")
#         return

#     # 3. 配置绘图选项
#     # 这一步是为了让生成的 SVG 更好看（可选）
#     draw_options = Draw.rdMolDraw2D.MolDrawOptions()
#     draw_options.addAtomIndices = False
#     draw_options.setBackgroundColour((1, 1, 1))
#     # 可以在这里设置高亮颜色，默认是浅红色

#     # 4. 生成网格 SVG 字符串
#     # MolsToGridImage 当 useSVG=True 时，返回的是 SVG 文本字符串
#     svg_grid = Draw.MolsToGridImage(
#         mols,
#         molsPerRow=mols_per_row,
#         subImgSize=(450, 450),        # 环肽较大，建议子图尺寸设大一点
#         legends=legends,
#         highlightAtomLists=highlight_lists,  # 【关键】传入高亮列表
#         useSVG=True,                  # 【关键】输出 SVG 格式
#         drawOptions=draw_options      # 应用绘图选项
#     )

#     # 5. 保存到文件
#     output_file = os.path.join(output_dir, output_file)
#     try:
#         with open(output_file, "w") as f:
#             f.write(svg_grid)
#         print(f"✅ 成功保存高亮网格图至: {output_file}")
#     except Exception as e:
#         print(f"保存文件失败: {e}")

# # --- 运行 ---
# # 请确保路径正确
# draw_highlighted_grid_svg('./output/generated/lora/gen_results_head_to_tail_2000.csv', output_dir='./output/generated/lora/peptide_images', max_mols=16)
#############################################################################

#############################################################################
# 这部分是绘图加相似性检测部分
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import DataStructs

def get_canonical_smiles(smi):
    """辅助函数：将 SMILES 转换为标准形式，用于去重对比"""
    if not smi: return None
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) # 忽略立体异构可根据需求改为 True
    return None

def get_morgan_fp(mol):
    """辅助函数：计算 Morgan 指纹"""
    if not mol: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def analyze_novelty_and_similarity(gen_df, ref_csv_paths):
    """
    判断生成序列的新颖性和相似度。
    
    Args:
        gen_df: 包含生成分子的 DataFrame (需包含 'smiles' 列)
        ref_csv_paths: 一个包含参考文件路径的列表
    Returns:
        处理后的 gen_df (新增 is_duplicate, max_similarity, most_similar_smi 列)
    """
    print("--- 开始进行新颖性与相似度分析 ---")
    
    # 1. 准备参考数据集 (合并所有参考文件)
    ref_smiles_set = set()
    ref_fps = []
    ref_mols_smiles_list = [] # 用于查找最相似的具体分子

    print("正在加载参考数据集...")
    for path in ref_csv_paths:
        if os.path.exists(path):
            try:
                # 假设参考文件也有 'smiles' 或 'SMILES' 列
                temp_df = pd.read_csv(path)
                col_name = 'smiles' if 'smiles' in temp_df.columns else 'SMILES'
                
                # 提取并标准化
                for smi in temp_df[col_name].dropna():
                    can_smi = get_canonical_smiles(smi)
                    if can_smi:
                        if can_smi not in ref_smiles_set: # 避免重复计算
                            ref_smiles_set.add(can_smi)
                            mol = Chem.MolFromSmiles(can_smi)
                            fp = get_morgan_fp(mol)
                            if fp:
                                ref_fps.append(fp)
                                ref_mols_smiles_list.append(can_smi)
            except Exception as e:
                print(f"读取参考文件 {path} 警告: {e}")
        else:
            print(f"警告: 文件不存在 {path}")

    print(f"参考库中共有 {len(ref_fps)} 个有效唯一分子。")

    # 2. 准备生成数据的结果列
    is_duplicate_list = []
    max_sim_list = []
    
    print(f"正在分析 {len(gen_df)} 个生成分子...")
    
    # 3. 遍历生成的分子进行对比
    # 注意：如果参考库极大(>10万)，这里可能需要优化，比如使用 Faiss 或 RDKit 的 BulkTanimoto
    for idx, row in gen_df.iterrows():
        gen_smi = row['smiles']
        can_gen_smi = get_canonical_smiles(gen_smi)
        
        if not can_gen_smi:
            is_duplicate_list.append(False)
            max_sim_list.append(0.0)
            continue

        # A. 查重 (Exact Match)
        is_dup = can_gen_smi in ref_smiles_set
        is_duplicate_list.append(is_dup)

        # B. 相似度 (Similarity)
        gen_mol = Chem.MolFromSmiles(can_gen_smi)
        gen_fp = get_morgan_fp(gen_mol)
        
        if gen_fp and ref_fps:
            # 批量计算 Tanimoto 相似度
            sims = DataStructs.BulkTanimotoSimilarity(gen_fp, ref_fps)
            max_sim = max(sims)
        else:
            max_sim = 0.0
        
        max_sim_list.append(max_sim)

    # 4. 将结果写回 DataFrame
    gen_df['is_in_database'] = is_duplicate_list
    gen_df['max_similarity'] = max_sim_list
    
    return gen_df

def draw_highlighted_grid_svg_with_check(csv_path, ref_paths, output_dir, output_file="cyclic_peptides_checked.svg", mols_per_row=4, max_mols=20):
    """
    读取生成文件 -> 查重/相似度分析 -> 绘图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取生成数据
    try:
        df = pd.read_csv(csv_path)
        # 过滤有效分子
        if 'is_valid' in df.columns:
            df = df[df['is_valid'] == True].copy()
        else:
            df = df.copy() # 如果没有 is_valid 列，假设全部需要处理
            
        print(f"读取到 {len(df)} 个有效生成分子。")
    except Exception as e:
        print(f"读取生成文件失败: {e}")
        return

    # --- 新增步骤：进行查重和相似度分析 ---
    # 为了绘图速度，我们只分析我们要画的那前 max_mols 个分子，或者分析全部
    # 这里为了演示，我们先切片再分析（节省时间），如果需要统计信息建议分析整个 df
    df_to_draw = df.head(max_mols).copy()
    
    # 调用分析函数
    df_analyzed = analyze_novelty_and_similarity(df_to_draw, ref_paths)
    # ------------------------------------

    mols = []
    highlight_lists = []
    legends = []

    print(f"正在绘制前 {len(df_analyzed)} 个分子...")

    for i, row in df_analyzed.iterrows():
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        if not mol: continue

        # A. 2D 坐标
        AllChem.Compute2DCoords(mol)

        # B. 大环高亮逻辑
        sssr = Chem.GetSymmSSSR(mol)
        current_mol_highlights = set()
        for ring in sssr:
            if len(ring) >= 10: 
                current_mol_highlights.update(ring)
        highlight_lists.append(list(current_mol_highlights))
        
        # C. 构造图例 (Legend)
        # 显示格式：Index | Sim: 0.85 | New/Exist
        sim_score = row['max_similarity']
        is_exist = row['is_in_database']
        status = "EXIST" if is_exist else "NEW"
        
        # 格式化图例文字
        legend_text = f"#{i} {status} (Sim:{sim_score:.2f})"
        legends.append(legend_text)
        
        mols.append(mol)

    if not mols:
        print("没有有效的分子可绘制。")
        return

    # 3. 绘图
    draw_options = Draw.rdMolDraw2D.MolDrawOptions()
    draw_options.addAtomIndices = False
    draw_options.setBackgroundColour((1, 1, 1))
    # 可以根据是否重复改变图例颜色等（需高级操作），这里仅改文字

    svg_grid = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(450, 450),
        legends=legends,
        highlightAtomLists=highlight_lists,
        useSVG=True,
        drawOptions=draw_options
    )

    # 4. 保存
    output_full_path = os.path.join(output_dir, output_file)
    try:
        with open(output_full_path, "w") as f:
            f.write(svg_grid)
        print(f"✅ 成功保存带相似度检测的网格图至: {output_full_path}")
        
        # 可选：同时也把分析结果保存为 CSV，方便查看
        analysis_csv = os.path.join(output_dir, "analysis_results.csv")
        df_analyzed.to_csv(analysis_csv, index=False)
        print(f"✅ 分析数据已保存至: {analysis_csv}")
        
    except Exception as e:
        print(f"保存文件失败: {e}")

# --- 运行设置 ---

# 生成文件路径
gen_csv_path = './output/generated/lora/gen_results_head_to_tail_2000.csv'

# 参考文件列表 (训练集和过滤集)
reference_files = [
    './data/restored_train.csv',
    'data/restored_validation.csv'
]

# 输出目录
out_dir = './output/generated/lora/peptide_images'

# 执行
draw_highlighted_grid_svg_with_check(
    csv_path=gen_csv_path, 
    ref_paths=reference_files, 
    output_dir=out_dir, 
    max_mols=16  # 为了测试速度，先设小一点
)