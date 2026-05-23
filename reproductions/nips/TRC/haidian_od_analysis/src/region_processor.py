"""
海淀区区域处理模块
功能：读取Shapefile文件，创建区域映射，进行空间匹配
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')


class RegionProcessor:
    """区域处理器"""
    
    def __init__(self, shapefile_path, region_mapping_path):
        """
        初始化区域处理器
        
        Args:
            shapefile_path: shapefile文件路径
            region_mapping_path: 区域映射表路径
        """
        self.shapefile_path = shapefile_path
        self.region_mapping_path = region_mapping_path
        self.regions_gdf = None
        self.region_mapping = None
        
    def load_regions(self):
        """加载区域边界数据"""
        print(f"正在加载区域边界文件: {self.shapefile_path}")
        try:
            # 尝试不同的编码方式
            try:
                self.regions_gdf = gpd.read_file(self.shapefile_path, encoding='utf-8')
            except UnicodeDecodeError:
                # 尝试GBK编码（中文Windows常用）
                try:
                    self.regions_gdf = gpd.read_file(self.shapefile_path, encoding='gbk')
                except:
                    # 最后尝试不指定编码
                    self.regions_gdf = gpd.read_file(self.shapefile_path)
            
            print(f"成功加载 {len(self.regions_gdf)} 个区域")
            print(f"区域数据列: {self.regions_gdf.columns.tolist()}")
            
            # 确保使用WGS84坐标系统
            if self.regions_gdf.crs is None:
                self.regions_gdf.set_crs('EPSG:4326', inplace=True)
            elif self.regions_gdf.crs != 'EPSG:4326':
                self.regions_gdf = self.regions_gdf.to_crs('EPSG:4326')
            
            return self.regions_gdf
        except Exception as e:
            print(f"加载区域边界文件失败: {e}")
            raise
    
    def load_region_mapping(self):
        """加载区域映射表"""
        print(f"正在加载区域映射表: {self.region_mapping_path}")
        try:
            self.region_mapping = pd.read_csv(self.region_mapping_path)
            print(f"成功加载 {len(self.region_mapping)} 个区域映射")
            print(f"映射表列: {self.region_mapping.columns.tolist()}")
            return self.region_mapping
        except Exception as e:
            print(f"加载区域映射表失败: {e}")
            raise
    
    def match_shapefile_to_mapping(self):
        """
        将Shapefile数据与区域映射表进行匹配
        返回合并后的GeoDataFrame
        """
        if self.regions_gdf is None:
            self.load_regions()
        if self.region_mapping is None:
            self.load_region_mapping()
        
        print("\n开始匹配Shapefile与区域映射表...")
        
        # 检查shapefile中的可用字段
        print(f"Shapefile字段: {self.regions_gdf.columns.tolist()}")
        print(f"前几行数据:\n{self.regions_gdf.head()}")
        
        # 尝试找到区域代码字段（可能是NAME, CODE, ADCODE等）
        # 这里需要根据实际shapefile字段调整
        merged_gdf = self.regions_gdf.copy()
        
        # 为每个区域分配region_id
        # 这里需要根据实际数据进行匹配逻辑
        # 暂时使用索引作为临时ID
        merged_gdf['region_id'] = range(1, len(merged_gdf) + 1)
        
        return merged_gdf
    
    def point_to_region(self, lon, lat):
        """
        将经纬度坐标映射到区域ID
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            region_id: 区域ID，如果不在任何区域内则返回None
        """
        if self.regions_gdf is None:
            raise ValueError("请先加载区域数据")
        
        point = Point(lon, lat)
        
        # 检查点在哪个区域内
        for idx, region in self.regions_gdf.iterrows():
            if region.geometry.contains(point):
                # 如果有region_id字段则返回，否则返回索引+1
                if 'region_id' in region:
                    return region['region_id']
                else:
                    return idx + 1
        
        return None
    
    def batch_points_to_regions(self, points_df, lon_col='longitude', lat_col='latitude'):
        """
        批量将点数据映射到区域
        
        Args:
            points_df: 包含经纬度的DataFrame
            lon_col: 经度列名
            lat_col: 纬度列名
            
        Returns:
            添加了region_id列的DataFrame
        """
        print(f"开始批量映射 {len(points_df)} 个点到区域...")
        
        if self.regions_gdf is None:
            raise ValueError("请先加载区域数据")
        
        # 创建点的GeoDataFrame
        geometry = [Point(xy) for xy in zip(points_df[lon_col], points_df[lat_col])]
        points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs='EPSG:4326')
        
        # 空间连接
        points_with_regions = gpd.sjoin(
            points_gdf, 
            self.regions_gdf[['geometry', 'region_id']] if 'region_id' in self.regions_gdf.columns 
            else self.regions_gdf.assign(region_id=range(1, len(self.regions_gdf) + 1))[['geometry', 'region_id']],
            how='left',
            predicate='within'
        )
        
        # 移除geometry列，返回普通DataFrame
        result_df = pd.DataFrame(points_with_regions.drop(columns='geometry'))
        
        # 统计匹配情况
        matched = result_df['region_id'].notna().sum()
        total = len(result_df)
        print(f"匹配成功: {matched}/{total} ({matched/total*100:.2f}%)")
        
        return result_df
    
    def get_region_info(self, region_id):
        """获取区域详细信息"""
        if self.region_mapping is None:
            self.load_region_mapping()
        
        region_info = self.region_mapping[self.region_mapping['region_id'] == region_id]
        if len(region_info) > 0:
            return region_info.iloc[0].to_dict()
        else:
            return None
    
    def export_region_geojson(self, output_path):
        """导出区域边界为GeoJSON格式"""
        if self.regions_gdf is None:
            raise ValueError("请先加载区域数据")
        
        self.regions_gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')
        print(f"区域边界已导出到: {output_path}")


if __name__ == "__main__":
    # 测试代码
    import sys
    
    shapefile_path = "/data/alice/cjtest/TRC/海淀区边界_110108_Shapefile_(poi86.com)/110108.shp"
    mapping_path = "/data/alice/cjtest/TRC/haidian_od_analysis/config/region_mapping.csv"
    
    processor = RegionProcessor(shapefile_path, mapping_path)
    processor.load_regions()
    processor.load_region_mapping()
    
    # 测试点映射
    test_lon, test_lat = 116.3, 39.99  # 海淀区某个点
    region_id = processor.point_to_region(test_lon, test_lat)
    print(f"\n测试点 ({test_lon}, {test_lat}) 属于区域ID: {region_id}")
    
    if region_id:
        info = processor.get_region_info(region_id)
        print(f"区域信息: {info}")
